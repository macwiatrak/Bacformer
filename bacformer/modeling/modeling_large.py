"""Bacformer Large Model Implementation.

This module contains the implementation of the Bacformer Large model, a transformer-based
architecture for bacterial genomics and protein sequence modeling. The model features:
- Rotary Position Embeddings (RoPE) for handling long sequences
- SwiGLU activation functions for improved performance
- Multi-head attention with optional attention weight output
- Gradient checkpointing support for memory-efficient training
- Masked genome modeling (MGM) pretraining objective
The architecture follows the ESM-C 300M geometry with 30 layers, 960-dimensional
hidden states, and 15 attention heads.
Link: https://github.com/evolutionaryscale/esm/blob/main/esm/models/esmc.py
Classes:
    BacformerModelOutput: Output dataclass for model predictions.
    SwiGLU: SwiGLU activation function module.
    RotaryEmbedding: Rotary position embedding implementation.
    RotaryMultiHeadAttention: Multi-head attention with rotary embeddings.
    TransformerBlock: Single transformer encoder block.
    BacformerLargeTransformerBlock: Full transformer encoder stack.
    BacformerPooler: Pooling layer for sequence-level representations.
    BacformerLargeEmbeddings: Input embedding layer with contig encoding.
    BacformerLargePreTrainedModel: Base class for pretrained models.
    BacformerLargeModel: Main Bacformer encoder model.
    BacformerLargeForMaskedGM: Bacformer with masked genome modeling head.
Example:
    >>> config = BacformerLargeConfig()
    >>> model = BacformerLargeModel(config)
    >>> # protein_embeddings shape: (batch_size, seq_len, hidden_size)
    >>> outputs = model(protein_embeddings)
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

SPECIAL_TOKENS_DICT = {
    "PAD": 0,
    "MASK": 1,
    "CLS": 2,
    "SEP": 3,
    "PROT_EMB": 4,
    "END": 5,
}


class BacformerLargeConfig(PretrainedConfig):
    """Configuration class for Bacformer Large model.

    This class stores all the configuration parameters needed to instantiate a
    Bacformer Large model. It inherits from `PretrainedConfig` and can be used
    to control the model architecture and behavior.
    The Bacformer Large model is a 30-layer transformer with 960-dimensional
    hidden states and 15 attention heads by default, designed for bacterial
    genome modeling tasks.
    Args:
        num_hidden_layers (int, optional): Number of transformer layers. Defaults to 30.
        num_attention_heads (int, optional): Number of attention heads per layer. Defaults to 15.
        hidden_size (int, optional): Dimensionality of hidden states. Defaults to 960.
        hidden_dropout_prob (float, optional): Dropout probability for hidden layers. Defaults to 0.1.
        final_layer_dropout_prob (float, optional): Dropout probability for final classification layer. Defaults to 0.1.
        attention_probs_dropout_prob (float, optional): Dropout probability for attention weights. Defaults to 0.1.
        max_position_embeddings (int, optional): Maximum sequence length the model can handle. Defaults to 6000.
        max_token_type_embeddings (int, optional): Maximum number of token types. Defaults to 1000.
        layer_norm_eps (float, optional): Epsilon for layer normalization. Defaults to 1e-12.
        initializer_range (float, optional): Standard deviation for weight initialization. Defaults to 0.02.
        pad_token_id (int, optional): Token ID for padding. Defaults to 0.
        mask_token_id (int, optional): Token ID for masked tokens. Defaults to 1.
        prot_emb_token_id (int, optional): Token ID for protein embeddings. Defaults to 4.
        cls_token_id (int, optional): Token ID for classification token. Defaults to 5.
        end_token_id (int, optional): Token ID for end of sequence. Defaults to 5.
        num_special_tokens (int, optional): Number of special tokens. Defaults to 6.
        protein_clusters_vocab_size (int, optional): Size of protein cluster vocabulary. Defaults to 50001.
        num_labels (int, optional): Number of labels for classification tasks. Defaults to 1.
        is_causal_gm (bool, optional): Whether to use causal attention for genome modeling. Defaults to False.
        return_dict (bool, optional): Whether to return a ModelOutput instead of tuple. Defaults to False.
        return_attn_weights (bool, optional): Whether to return attention weights. Defaults to False.
        gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        problem_type (str, optional): Type of problem for classification head.
            One of "regression", "binary_classification", "single_label_classification",
            or "multi_label_classification". Defaults to "single_label_classification".
        **kwargs: Additional arguments passed to `PretrainedConfig`.

    Attributes
    ----------
        model_type (str): The model type identifier, set to "bacformer".
    Example:
        >>> config = BacformerLargeConfig(
        ...     num_hidden_layers=24,
        ...     hidden_size=768,
        ...     num_attention_heads=12,
        ... )
        >>> config.hidden_size
        768

        >>> # Default config uses 960-dimensional hidden states
        >>> default_config = BacformerLargeConfig()
        >>> default_config.hidden_size
        960
    """

    model_type = "bacformer"

    def __init__(
        self,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 15,
        hidden_size: int = 960,
        hidden_dropout_prob: float = 0.1,
        final_layer_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 6000,
        max_token_type_embeddings: int = 1000,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        pad_token_id: int = SPECIAL_TOKENS_DICT["PAD"],
        mask_token_id: int = SPECIAL_TOKENS_DICT["MASK"],
        prot_emb_token_id: int = SPECIAL_TOKENS_DICT["PROT_EMB"],
        cls_token_id: int = SPECIAL_TOKENS_DICT["END"],
        end_token_id: int = SPECIAL_TOKENS_DICT["END"],
        num_special_tokens: int = len(SPECIAL_TOKENS_DICT),
        protein_clusters_vocab_size: int = 50001,  # equal to the nr of protein clusters + 1
        num_labels: int = 1,  # for downstream tasks
        is_causal_gm: bool = False,
        return_dict: bool = False,
        return_attn_weights: bool = False,
        gradient_checkpointing: bool = False,
        problem_type: Literal[
            "regression", "binary_classification", "single_label_classification", "multi_label_classification"
        ] = "single_label_classification",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.final_layer_dropout_prob = final_layer_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_token_type_embeddings = max_token_type_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.prot_emb_token_id = prot_emb_token_id
        self.end_token_id = end_token_id
        self.num_special_tokens = num_special_tokens
        self.protein_clusters_vocab_size = protein_clusters_vocab_size
        self.num_labels = num_labels
        self.is_causal_gm = is_causal_gm
        self.return_dict = return_dict
        self.return_attn_weights = return_attn_weights
        self.problem_type = problem_type
        self.gradient_checkpointing = gradient_checkpointing


@dataclass
class BacformerModelOutput(ModelOutput):
    """Output dataclass for Bacformer model predictions.

    This class extends HuggingFace's ModelOutput to provide structured outputs
    from the Bacformer model, supporting both tuple-style and attribute-style access.

    Attributes
    ----------
        loss (torch.FloatTensor, optional): Loss value when labels are provided.
            Shape: scalar.
        logits (torch.FloatTensor): Prediction logits from the classification head.
            Shape: (batch_size, sequence_length, vocab_size) for MLM or
            (batch_size, num_labels) for classification.
        last_hidden_state (torch.FloatTensor, optional): Hidden states from the
            last transformer layer. Shape: (batch_size, sequence_length, hidden_size).
        attentions (list[torch.FloatTensor], optional): Attention weights from all
            layers when `return_attn_weights=True`. Each tensor has shape:
            (batch_size, num_heads, sequence_length, sequence_length).
        pooler_output (torch.FloatTensor, optional): Pooled representation of the
            sequence. Shape: (batch_size, hidden_size).
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor | None = None
    attentions: list[torch.FloatTensor] | None = None
    pooler_output: torch.FloatTensor | None = None


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    """Compute corrected intermediate dimension for SwiGLU feedforward network.

    SwiGLU requires the intermediate dimension to be a multiple of 256 for
    efficient GPU computation. This function computes the corrected dimension.
    Args:
        expansion_ratio: Ratio to expand the model dimension.
        d_model: Model hidden dimension.

    Returns
    -------
        Corrected intermediate dimension rounded up to nearest multiple of 256.
    """
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    SwiGLU (Swish-Gated Linear Unit) is an activation function that combines
    the Swish activation with a gating mechanism. It splits the input into
    two halves, applies Swish (SiLU) to the first half, and uses the second
    half as a gate.
    Reference:
        Shazeer, N. (2020). GLU Variants Improve Transformer.
        https://arxiv.org/abs/2002.05202
    Example:
        >>> swiglu = SwiGLU()
        >>> x = torch.randn(2, 10, 512)  # Input with doubled last dim
        >>> output = swiglu(x)  # Output shape: (2, 10, 256)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.

        Args:
            x: Input tensor with last dimension to be split in half.
                Shape: (..., 2 * hidden_dim).

        Returns
        -------
            Activated tensor with halved last dimension.
            Shape: (..., hidden_dim).
        """
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float) -> nn.Sequential:
    """Create SwiGLU feedforward network with layer normalization.

    Constructs a feedforward network consisting of:
    1. Layer normalization
    2. Linear projection to expanded dimension (2x for gating)
    3. SwiGLU activation
    4. Linear projection back to model dimension
    Args:
        d_model: Model hidden dimension.
        expansion_ratio: Ratio to expand the intermediate dimension.
            Default in transformer is typically 8/3 for SwiGLU.

    Returns
    -------
        Sequential module containing the feedforward network.
    """
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=False),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=False),
    )


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input tensor.

    This is a helper function for applying rotary position embeddings.
    It rotates the tensor by splitting it in half and swapping/negating.
    Args:
        x: Input tensor to rotate. Shape: (..., dim).
        interleaved: If True, uses interleaved layout where even/odd
            dimensions are grouped. If False, uses split layout.

    Returns
    -------
        Rotated tensor with same shape as input.
    """
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    _inplace: bool = False,
) -> torch.Tensor:
    """Apply rotary embeddings to input based on cos and sin."""
    # taken from ESM-C model, https://huggingface.co/Synthyra/ESMplusplus_small
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    cos = repeat(cos, "s d -> s 1 (2 d)")
    sin = repeat(sin, "s d -> s 1 (2 d)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class RotaryEmbedding(torch.nn.Module):
    """Rotary position embeddings (RoFormer)."""

    # taken from ESM-C model, https://huggingface.co/Synthyra/ESMplusplus_small

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: float | None = None,
        scaling_factor: float = 1.0,
        pos_idx_in_fp32: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.device = device

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the embedding."""
        inv_freq = self._compute_inv_freq(self.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        arange = torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32)
        scale = (arange + 0.4 * self.dim) / (1.4 * self.dim) if self.scale_base is not None else None
        self.register_buffer("scale", scale)

    def _compute_inv_freq(self, device: torch.device | None = None) -> torch.Tensor:
        """Compute inverse frequency bands."""
        return 1 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(self, seqlen: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """Update the cached cosine and sine values."""
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)

            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys (sequential positions)."""
        self._update_cos_sin_cache(q.shape[1], device=q.device, dtype=q.dtype)
        assert self._cos_cached is not None
        assert self._sin_cached is not None
        if self.scale is None:
            return (
                apply_rotary_emb_torch(
                    q,
                    self._cos_cached,
                    self._sin_cached,
                    self.interleaved,
                    True,  # inplace=True
                ),
                apply_rotary_emb_torch(
                    k,
                    self._cos_cached,
                    self._sin_cached,
                    self.interleaved,
                    True,  # inplace=True
                ),
            )  # type: ignore
        else:
            raise AssertionError()


# ====== Attention with optional distance-aware RoPE ======
class RotaryMultiHeadAttention(nn.Module):
    """Multi-head attention with rotary embeddings."""

    # taken from ESM-C model, https://huggingface.co/Synthyra/ESMplusplus_small

    def __init__(self, d_model: int, n_heads: int, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=False))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_ln = nn.LayerNorm(d_model, bias=False)
        self.k_ln = nn.LayerNorm(d_model, bias=False)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbedding(d_model // n_heads, base=base)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys."""
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        x: torch.Tensor,  # (B,S,D)
        attention_mask: torch.Tensor | None = None,  # (B,S) bool or 0/1
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head attention with rotary embeddings.

        Args:
            x: Input tensor of shape (B, S, D)
            attention_mask: Optional attention mask of shape (B, S) with True for valid tokens and False for padding.
            return_attn_weights: Whether to return attention weights for debugging and analysis.
        """
        attn_weights = None
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = self.q_ln(query_BLD).to(query_BLD.dtype), self.k_ln(key_BLD).to(query_BLD.dtype)

        # [CHANGED] apply RoPE (sequential or distance-aware)
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        query_BHLD, key_BHLD, value_BHLD = map(self.reshaper, (query_BLD, key_BLD, value_BLD))

        attn_mask = None
        if attention_mask is not None:
            # make it broadcastable to (B, H, L, S)
            attn_mask = attention_mask[:, None, None, :].to(torch.bool)  # (B,1,1,S)

        if return_attn_weights:
            b, h, l, d = query_BHLD.shape
            scale = 1 / math.sqrt(d)
            attn_scores = torch.matmul(query_BHLD, key_BHLD.transpose(-2, -1)) * scale  # (B,H,L,S)

            if attn_mask is not None:
                # mask out padding positions in keys (across S)
                attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))

            attn_weights = F.softmax(attn_scores, dim=-1)
            context_BHLD = torch.matmul(attn_weights, value_BHLD)
        else:
            # PyTorch SDPA: attn_mask must be broadcastable to (B,H,L,S)
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD,
                key_BHLD,
                value_BHLD,
                attn_mask=attn_mask,  # (B,1,1,S) -> broadcasts to (B,H,L,S)
                # is_causal=False  # leave default unless you want causal masking
            )

        context_BLD = rearrange(context_BHLD, "b h s d -> b s (h d)")
        output = self.out_proj(context_BLD)
        return output, attn_weights


# ----------  B. Transformer block  ------------------------------------
class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm architecture.

    Consists of:
    1. Pre-LayerNorm + Multi-head attention with RoPE + residual connection
    2. Pre-LayerNorm + SwiGLU FFN + residual connection
    Uses residue scaling for deep network training stability.
    Args:
        hidden_size: Hidden dimension of the transformer.
        n_heads: Number of attention heads.
        dropout: Dropout probability. Defaults to 0.0.
        residue_scaling_factor: Factor to scale residual connections by,
            computed as sqrt(num_layers / 36) for training stability.
        expansion_ratio: FFN expansion ratio. Defaults to 8/3 for SwiGLU.

    Attributes
    ----------
        norm: Pre-attention layer normalization.
        attn: Multi-head attention with rotary embeddings.
        ffn: SwiGLU feedforward network.
        scaling_factor: Residue scaling factor.
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: float = 0.0,
        residue_scaling_factor: float = 1.0,
        expansion_ratio: float = 8 / 3,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, bias=False)
        self.attn = RotaryMultiHeadAttention(hidden_size, n_heads)
        self.ffn = swiglu_ln_ffn(hidden_size, expansion_ratio)
        self.scaling_factor = residue_scaling_factor
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        is_causal: bool = False,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor. Shape: (batch_size, seq_len, hidden_dim).
            mask: Optional attention mask. Shape: (batch_size, seq_len).
            is_causal: Whether to apply causal masking. Currently unused
                but kept for API compatibility.
            return_attn_weights: Whether to return attention weights.

        Returns
        -------
            Tuple of (output, attention_weights) where output has the same
            shape as input and attention_weights is None or has shape
            (batch_size, num_heads, seq_len, seq_len).
        """
        attn_output, attn_weights = self.attn(self.norm(x), mask, return_attn_weights)
        x = x + self.dropout(attn_output) / self.scaling_factor
        x = x + self.dropout(self.ffn(x)) / self.scaling_factor
        return x, attn_weights


class BacformerLargeTransformerBlock(nn.Module):
    """Full transformer encoder stack for Bacformer Large.

    Implements a stack of transformer blocks following the ESM-C 300M geometry:
    - 30 layers by default
    - 960-dimensional hidden states
    - 15 attention heads
    - SwiGLU activation with 8/3 expansion ratio
    Supports gradient checkpointing for memory-efficient training of deep models.
    Args:
        hidden_size: Hidden dimension. Defaults to 960.
        n_heads: Number of attention heads. Defaults to 15.
        n_layers: Number of transformer layers. Defaults to 30.
        dropout: Dropout probability. Defaults to 0.0.
        expansion_ratio: FFN expansion ratio. Defaults to 8/3.

    Attributes
    ----------
        gradient_checkpointing: Whether to use gradient checkpointing.
        layers: ModuleList of TransformerBlock instances.
        final_ln: Final layer normalization after all transformer blocks.
    """

    def __init__(
        self,
        hidden_size: int = 960,
        n_heads: int = 15,
        n_layers: int = 30,
        dropout: float = 0.0,
        expansion_ratio: float = 8 / 3,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    n_heads=n_heads,
                    dropout=dropout,
                    residue_scaling_factor=math.sqrt(n_layers / 36),
                    expansion_ratio=expansion_ratio,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(hidden_size, bias=False)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,  # (B,S,D)
        attention_mask: torch.Tensor = None,  # (B,S)  1/0
        is_causal: bool = False,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Forward pass through all transformer layers.

        Args:
            x: Input tensor. Shape: (batch_size, seq_len, hidden_dim).
            attention_mask: Optional attention mask where True/1 indicates
                tokens to attend to. Shape: (batch_size, seq_len).
            is_causal: Whether to apply causal masking (passed to blocks).
            return_attn_weights: Whether to return attention weights from
                all layers.

        Returns
        -------
            Tuple of:
                - Output tensor with shape (batch_size, seq_len, hidden_dim).
                - List of attention weights from each layer (None if not requested).
        """
        attn_weights_arr = []
        for block in self.layers:
            if self.gradient_checkpointing and self.training:
                x, attn_weights = checkpoint(
                    block.__call__,
                    x,
                    attention_mask,
                    is_causal,
                    return_attn_weights,
                    use_reentrant=False,
                )
            else:
                x, attn_weights = block(x, attention_mask, is_causal, return_attn_weights)
            attn_weights_arr.append(attn_weights)
        return self.final_ln(x), attn_weights_arr


class BacformerPooler(nn.Module):
    """Pooling layer for obtaining sequence-level representations.

    Computes a fixed-size representation of the entire sequence by:
    1. Computing mean of non-padding token hidden states
    2. Applying a dense layer
    3. Applying tanh activation
    Args:
        config: Bacformer configuration object.

    Attributes
    ----------
        dense: Linear projection layer.
        activation: Tanh activation function.
    """

    def __init__(self, config: BacformerLargeConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Pool hidden states to a single sequence representation.

        Args:
            hidden_states: Hidden states from the encoder.
                Shape: (batch_size, seq_len, hidden_dim).
            padding_mask: Optional mask indicating valid (non-padding) tokens
                where 1 indicates valid tokens. Shape: (batch_size, seq_len).

        Returns
        -------
            Pooled representation. Shape: (batch_size, hidden_dim).
        """
        # We "pool" the model by taking the mean of non-padding tokens
        padding_mask = padding_mask.to(hidden_states.device) if padding_mask is not None else None
        if padding_mask is not None:
            mean_hidden_states = torch.einsum("ijk,ij->ik", hidden_states, padding_mask) / padding_mask.sum(
                1
            ).unsqueeze(1)
        else:
            mean_hidden_states = hidden_states.mean(dim=1)
        pooled_output = self.dense(mean_hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BacformerLargeEmbeddings(nn.Module):
    """Input embedding layer for Bacformer Large.

    Combines protein embeddings with contig ID embedding and handles
    masked language modeling (MLM) token replacement.
    The embedding process:
    1. Project protein embeddings to model dimension
    2. Replace masked positions with learned mask embedding
    3. Add contig ID embeddings (proteins belonging to the same contig share the same ID)
    4. Apply layer normalization and dropout
    Args:
        config: Bacformer configuration object.

    Attributes
    ----------
        linear: Projection layer for protein embeddings.
        contig_embeddings: Embedding layer for contig identifiers.
        mask_embed: Learned embedding for masked positions.
        LayerNorm: Layer normalization.
        dropout: Dropout layer.
    """

    def __init__(self, config: BacformerLargeConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.contig_embeddings = nn.Embedding(
            num_embeddings=config.max_token_type_embeddings + 1,
            embedding_dim=config.hidden_size,
            padding_idx=config.max_token_type_embeddings,
        )

        self.mask_embed = nn.Parameter(torch.empty(config.hidden_size))
        nn.init.normal_(self.mask_embed, std=0.02)

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        contig_ids: torch.Tensor | None = None,
        mlm_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute input embeddings from protein representations.

        Args:
            protein_embeddings: Pre-computed protein embeddings.
                Shape: (batch_size, seq_len, hidden_dim).
            contig_ids: Contig identifiers (integers).
                Shape: (batch_size, seq_len).
            mlm_mask: Optional mask indicating positions to replace with mask embedding.
                Shape: (batch_size, seq_len).

        Returns
        -------
            Processed embeddings. Shape: (batch_size, seq_len, hidden_dim).
        """
        # 1) Project to model dim (non-leaf, safe for in-place row writes)
        x = self.linear(protein_embeddings)  # (B, S, D)

        # 2) Replace only masked rows with a learned vector — no big clone
        if mlm_mask is not None:
            m = mlm_mask.to(torch.bool)
            if m.any():
                x2d = x.view(-1, x.size(-1))  # (B*S, D)
                idx = m.view(-1).nonzero(as_tuple=False).squeeze(1)  # (N_mask,)
                src = self.mask_embed.to(dtype=x2d.dtype, device=x2d.device).expand(idx.numel(), -1)  # (N_mask, D)
                x2d.index_copy_(0, idx, src)  # in-place, grads flow to mask_embed

        if contig_ids is not None:
            x = x + self.contig_embeddings(contig_ids.long())

        # 5) Norm + dropout
        x = self.norm1(x)
        x = self.dropout(x)
        return x


def ClassificationHead(d_model: int, output_dim: int, hidden_dim: int | None = None, dropout: float = 0.1) -> nn.Module:
    """Create a classification head for downstream tasks.

    Constructs a two-layer MLP with GELU activation and layer normalization:
    1. Dropout
    2. Linear(d_model -> hidden_dim)
    3. GELU activation
    4. LayerNorm
    5. Linear(hidden_dim -> output_dim)
    Args:
        d_model: Input dimension (model hidden size).
        output_dim: Output dimension (number of classes or 1 for regression).
        hidden_dim: Hidden dimension of the MLP. Defaults to d_model.
        dropout: Dropout probability. Defaults to 0.1.

    Returns
    -------
        Sequential module containing the classification head.
    Example:
        >>> head = ClassificationHead(d_model=768, output_dim=2, dropout=0.1)
        >>> x = torch.randn(32, 768)  # pooled representations
        >>> logits = head(x)  # shape: (32, 2)
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


class BacformerLargePreTrainedModel(PreTrainedModel):
    """Abstract base class for Bacformer Large pretrained models.

    Provides common functionality for all Bacformer models:
    - Weight initialization
    - Gradient checkpointing support
    - Loading/saving pretrained weights
    - HuggingFace Hub integration
    This class should not be instantiated directly. Use derived classes
    like `BacformerLargeModel` or `BacformerLargeForMaskedGM` instead.

    Attributes
    ----------
        config_class: The configuration class for this model (BacformerLargeConfig).
        base_model_prefix: Prefix for the base model in state dict ("bacformer_large").
        supports_gradient_checkpointing: Whether gradient checkpointing is supported (True).
    """

    config_class = BacformerLargeConfig
    base_model_prefix = "bacformer"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize model weights.

        Applies the following initialization scheme:
        - Linear layers: Normal distribution with std=config.initializer_range
        - Embeddings: Normal distribution with std=config.initializer_range
        - LayerNorm: Weight=1.0, bias=0.0
        - RotaryEmbedding: Non-persistent `inv_freq` buffer rebuilt analytically
        Args:
            module: The module to initialize.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, RotaryEmbedding):
            # Rebuild non-persistent RoPE buffers after meta-device loading (#33).
            module.inv_freq.copy_(module._compute_inv_freq(module.inv_freq.device))


class BacformerLargeModel(BacformerLargePreTrainedModel):
    """Bacformer Large encoder model.

    The base Bacformer model that outputs raw hidden states from the transformer
    encoder. Can optionally include a pooling layer for sequence-level tasks.
    This model takes protein embeddings (pre-computed from sequences) as input
    and produces contextualized representations.
    Args:
        config: Model configuration.
        add_pooling_layer: Whether to add a pooling layer on top of the encoder.
            Defaults to False.

    Attributes
    ----------
        embeddings: Input embedding layer.
        encoder: Transformer encoder stack.
        pooler: Optional pooling layer (None if add_pooling_layer=False).
    Example:
        >>> config = BacformerLargeConfig()
        >>> model = BacformerLargeModel(config, add_pooling_layer=True)
        >>> protein_embeddings = torch.randn(2, 100, 920)  # (batch, seq_len, hidden)
        >>> outputs = model(protein_embeddings, return_dict=True)
        >>> last_hidden = outputs.last_hidden_state  # (2, 100, 920)
        >>> pooled = outputs.pooler_output  # (2, 920)
    """

    def __init__(self, config: BacformerLargeConfig, add_pooling_layer: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = BacformerLargeEmbeddings(config)
        self.encoder = BacformerLargeTransformerBlock(
            hidden_size=config.hidden_size,
            n_heads=config.num_attention_heads,
            n_layers=config.num_hidden_layers,
            dropout=config.hidden_dropout_prob,
        )

        self.pooler = BacformerPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        attention_mask: torch.Tensor = None,
        contig_ids: torch.Tensor | None = None,
        mlm_mask: torch.Tensor | None = None,
        return_attn_weights: bool = False,
        return_dict: bool | None = None,
        is_causal: bool = False,
    ) -> BacformerModelOutput | None:
        """Forward pass of the Bacformer encoder.

        Args:
            protein_embeddings: Pre-computed protein embeddings.
                Shape: (batch_size, seq_len, hidden_dim).
            attention_mask: Optional mask where 1 indicates valid tokens.
                Shape: (batch_size, seq_len).
            contig_ids: Contig identifiers (integers).
                Shape: (batch_size, seq_len).
            mlm_mask: Optional mask for masked language modeling positions.
                Shape: (batch_size, seq_len).
            return_attn_weights: Whether to return attention weights.
            return_dict: Whether to return a ModelOutput or tuple.
                Defaults to config.return_dict.
            is_causal: Whether to apply causal masking.

        Returns
        -------
            If return_dict=False: Tuple of (last_hidden_state, pooler_output).
            If return_dict=True: BacformerModelOutput with last_hidden_state,
                pooler_output, and optionally attentions.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # get embeddings
        protein_embeddings = self.embeddings(
            protein_embeddings=protein_embeddings,
            contig_ids=contig_ids,
            mlm_mask=mlm_mask,
        )

        last_hidden_state, attentions = self.encoder(
            x=protein_embeddings,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
            is_causal=is_causal,
            return_attn_weights=return_attn_weights,
        )
        pooler_output = (
            self.pooler(hidden_states=last_hidden_state, padding_mask=attention_mask)
            if self.pooler is not None
            else None
        )

        if not return_dict:
            return (last_hidden_state, pooler_output)

        return BacformerModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            attentions=attentions,
        )


class BacformerLargeForMaskedGM(BacformerLargePreTrainedModel):
    """Bacformer Large with masked genome modeling (MGM) head.

    This model is designed for pretraining using the masked genome modeling
    objective. It predicts protein cluster identities for masked positions
    in the genome sequence.
    The MGM head is a two-layer MLP that predicts over the protein cluster
    vocabulary from the hidden states of masked positions.
    Args:
        config: Model configuration.

    Attributes
    ----------
        bacformer: The base Bacformer encoder model.
        gm_head: Classification head for predicting protein clusters.
    Example:
        >>> config = BacformerLargeConfig(protein_clusters_vocab_size=50001)
        >>> model = BacformerLargeForMaskedGM(config)
        >>> protein_embeddings = torch.randn(2, 100, 920)
        >>> labels = torch.randint(0, 50001, (2, 100))
        >>> labels[labels != labels] = -100  # mask non-target positions
        >>> outputs = model(protein_embeddings, labels=labels, return_dict=True)
        >>> loss = outputs.loss
    """

    def __init__(self, config: BacformerLargeConfig):
        super().__init__(config)
        self.config = config

        self.bacformer = BacformerLargeModel(config, add_pooling_layer=False)
        self.gm_head = ClassificationHead(
            d_model=config.hidden_size,
            output_dim=config.protein_clusters_vocab_size,
            hidden_dim=config.hidden_size * 2,
            dropout=config.final_layer_dropout_prob,
        )

        # Initialize weights
        self.post_init()

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        attention_mask: torch.Tensor = None,
        contig_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        mlm_mask: torch.Tensor | None = None,
        return_attn_weights: bool = False,
        return_dict: bool | None = None,
        is_causal: bool = False,
    ) -> BacformerModelOutput | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional masked genome modeling loss computation.

        Args:
            protein_embeddings: Pre-computed protein embeddings.
                Shape: (batch_size, seq_len, hidden_dim).
            labels: Optional target protein cluster IDs. Use -100 for positions
                that should not contribute to the loss.
                Shape: (batch_size, seq_len).
            attention_mask: Optional mask where 1 indicates valid tokens.
                Shape: (batch_size, seq_len).
            contig_ids: Contig identifiers (integers).
                Shape: (batch_size, seq_len).
            mlm_mask: Optional mask for masked positions.
                Shape: (batch_size, seq_len).
            return_attn_weights: Whether to return attention weights.
            return_dict: Whether to return a ModelOutput or tuple.
                Defaults to config.return_dict.
            is_causal: Whether to apply causal masking.

        Returns
        -------
            If return_dict=False: Tuple of (loss, prediction_scores, hidden_states, ...).
            If return_dict=True: BacformerModelOutput with loss, logits,
                last_hidden_state, and optionally attentions.
        Note:
            When labels are provided, the model only computes predictions for
            masked positions (labels != -100) to improve efficiency.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # get Bacformer embeddings
        outputs = self.bacformer(
            protein_embeddings=protein_embeddings,
            attention_mask=attention_mask,
            contig_ids=contig_ids,
            mlm_mask=mlm_mask,
            return_attn_weights=return_attn_weights,
            is_causal=is_causal,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        loss = None
        if labels is not None:
            # to speed up the forward pass, let's only consider the masked tokens
            prediction_scores = self.gm_head(last_hidden_state[labels != -100])
            labels = labels.to(prediction_scores.device)

            # only considering the masked tokens
            labels = labels[labels != -100]
            if labels.numel() == 0:
                # Force a synchronized "zero loss" that still runs backward on all ranks
                # Uses a dependency on model outputs so autograd graph exists:
                loss = last_hidden_state.sum() * 0.0
            else:
                loss = F.cross_entropy(prediction_scores, labels)
        else:
            prediction_scores = self.gm_head(last_hidden_state)

        if not return_dict:
            return (
                loss,
                prediction_scores,
            ) + outputs

        return BacformerModelOutput(
            loss=loss,
            logits=prediction_scores,
            last_hidden_state=last_hidden_state,
            attentions=outputs.attentions,
        )


class BacformerGenomeClassificationHead(nn.Module):
    """Head for genome-level classification tasks.

    Pools sequence representations using attention mask and applies
    classification layers.
    Args:
        config: Bacformer configuration object.

    Attributes
    ----------
        norm: Layer normalization.
        dropout: Dropout layer.
        out_proj: Output projection to num_labels.
    """

    def __init__(self, config: BacformerLargeConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor, **kwargs):
        """Forward pass for the genome classification head.

        Args:
            features: Hidden states from encoder. Shape: (batch_size, seq_len, hidden_size).
            padding_mask: Mask indicating valid tokens. Shape: (batch_size, seq_len). Padded positions should be 0.

        Returns
        -------
            Classification logits. Shape: (batch_size, num_labels).
        """
        if padding_mask is not None:
            x = torch.einsum("ijk,ij->ik", features, padding_mask) / padding_mask.sum(1).unsqueeze(1)
        else:
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.norm(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BacformerProteinProteinInteractionHead(nn.Module):
    """Head for protein-protein interaction prediction at genome level.

    Args:
        in_features: Input feature dimension.
        bias: Whether to use bias in linear layer. Defaults to True.
        dropout: Dropout probability. Defaults to 0.1.

    Attributes
    ----------
        dropout: Dropout layer.
        linear: Output linear layer.
    """

    def __init__(self, in_features: int, bias: bool = True, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, 1, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for the PPI head.

        Args:
            hidden_states: Pooled representation of protein pair.
                Shape: (hidden_size,) or (batch_size, hidden_size).

        Returns
        -------
            Interaction score. Shape: scalar or (batch_size,).
        """
        return self.linear(self.dropout(hidden_states)).squeeze(-1)


class BacformerLargeForProteinClassification(BacformerLargePreTrainedModel):
    """Bacformer Large model with protein classification head.

    This model is designed for protein-level classification tasks where
    each protein in the genome receives a classification label.
    Args:
        config: Model configuration.

    Attributes
    ----------
        bacformer: The base Bacformer encoder model.
        dropout: Dropout layer.
        classifier: Linear classification head.
    Example:
        >>> config = BacformerLargeConfig(num_labels=2)
        >>> model = BacformerLargeForProteinClassification(config)
        >>> protein_embeddings = torch.randn(2, 100, 920)
        >>> labels = torch.randint(0, 2, (2, 100))
        >>> outputs = model(protein_embeddings, labels=labels, return_dict=True)
        >>> loss = outputs.loss
    """

    def __init__(self, config: BacformerLargeConfig):
        super().__init__(config)
        self.config = config

        self.bacformer = BacformerLargeModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor = None,
        contig_ids: torch.Tensor | None = None,
        return_attn_weights: bool = False,
        return_dict: bool | None = None,
        special_tokens_mask=None,  # for compatibility with Bacformer base 26M
    ) -> BacformerModelOutput | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for protein classification.

        Args:
            protein_embeddings: Pre-computed protein embeddings.
                Shape: (batch_size, seq_len, hidden_dim).
            labels: Optional target labels for each protein.
                Shape: (batch_size, seq_len) or (batch_size, seq_len, num_labels).
            attention_mask: Optional mask where 1 indicates valid tokens.
                Shape: (batch_size, seq_len).
            contig_ids: Optional contig identifiers.
                Shape: (batch_size, seq_len).
            return_attn_weights: Whether to return attention weights.
            return_dict: Whether to return a ModelOutput or tuple.
            special_tokens_mask: For compatibility with Bacformer 26M.

        Returns
        -------
            If return_dict=False: Tuple of (loss, None, logits).
            If return_dict=True: BacformerModelOutput with loss, logits,
                last_hidden_state, and optionally attentions.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # get Bacformer embeddings
        outputs = self.bacformer(
            protein_embeddings=protein_embeddings,
            attention_mask=attention_mask,
            contig_ids=contig_ids,
            return_attn_weights=return_attn_weights,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type == "regression":
                loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif (
                self.config.problem_type == "multi_label_classification"
                or self.config.problem_type == "binary_classification"
            ):
                # remove the -100 labels from loss computation
                mask = torch.ones_like(labels.view(-1)) - (labels.view(-1) == -100.0).float()
                loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1), labels.view(-1).type_as(logits), reduction="none"
                )
                loss = (loss * mask).sum() / mask.sum()

        if not return_dict:
            return (
                loss,
                None,
                logits,
            )

        return BacformerModelOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=last_hidden_state,
            attentions=outputs.attentions,
        )


class BacformerLargeForGenomeClassification(BacformerLargePreTrainedModel):
    """Bacformer Large model with genome classification head.

    This model is designed for genome-level classification tasks where
    the entire genome receives a single classification label.
    Args:
        config: Model configuration.

    Attributes
    ----------
        bacformer: The base Bacformer encoder model.
        classifier: Genome classification head with pooling.
    Example:
        >>> config = BacformerLargeConfig(num_labels=3)
        >>> model = BacformerLargeForGenomeClassification(config)
        >>> protein_embeddings = torch.randn(2, 100, 920)
        >>> labels = torch.tensor([0, 1])
        >>> outputs = model(protein_embeddings, labels=labels, return_dict=True)
        >>> loss = outputs.loss
    """

    def __init__(self, config: BacformerLargeConfig):
        super().__init__(config)
        self.config = config

        self.bacformer = BacformerLargeModel(config, add_pooling_layer=False)
        self.classifier = BacformerGenomeClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        contig_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        return_attn_weights: bool = None,
        return_dict: bool | None = None,
        special_tokens_mask=None,  # for compatibility with Bacformer 26M
    ) -> BacformerModelOutput | None:
        """Forward pass for genome classification.

        Args:
            protein_embeddings: Pre-computed protein embeddings.
                Shape: (batch_size, seq_len, hidden_dim).
            contig_ids: Optional contig identifiers.
                Shape: (batch_size, seq_len).
            attention_mask: Optional mask where 1 indicates valid tokens.
                Shape: (batch_size, seq_len).
            labels: Optional target labels for each genome.
                Shape: (batch_size,) or (batch_size, num_labels).
            return_attn_weights: Whether to return attention weights.
            return_dict: Whether to return a ModelOutput or tuple.
            special_tokens_mask: For compatibility with Bacformer 26M.

        Returns
        -------
            If return_dict=False: Tuple of (loss, None, logits).
            If return_dict=True: BacformerModelOutput with loss, logits,
                last_hidden_state, and optionally attentions.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        return_attn_weights = (
            return_attn_weights if return_attn_weights is not None else self.config.return_attn_weights
        )

        outputs = self.bacformer(
            protein_embeddings=protein_embeddings,
            contig_ids=contig_ids,
            attention_mask=attention_mask,
            return_attn_weights=return_attn_weights,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]
        logits = self.classifier(last_hidden_state, attention_mask)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type == "regression":
                loss = F.mse_loss(logits.view(-1), labels.view(-1))
            elif self.config.problem_type == "binary_classification":
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits))
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            return (
                loss,
                None,
                logits,
            )

        return BacformerModelOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )


class BacformerLargeForProteinProteinInteraction(BacformerLargePreTrainedModel):
    """Bacformer Large model for protein-protein interaction prediction.

    This model predicts whether two proteins in a genome interact with each other.
    It uses the hidden states of the two proteins and predicts an interaction score.
    Note:
        This model expects batch_size=1 as it processes protein pairs within a genome.
    Args:
        config: Model configuration.

    Attributes
    ----------
        bacformer: The base Bacformer encoder model.
        dropout: Dropout layer.
        dense: Dense transformation layer.
        ppi_head: Protein-protein interaction prediction head.
    Example:
        >>> config = BacformerLargeConfig()
        >>> model = BacformerLargeForProteinProteinInteraction(config)
        >>> protein_embeddings = torch.randn(1, 100, 920)
        >>> # labels: [protein1_idx, protein2_idx, interaction_label]
        >>> labels = torch.tensor([[5, 10, 1]])
        >>> outputs = model(protein_embeddings, labels=labels, return_dict=True)
    """

    def __init__(self, config: BacformerLargeConfig):
        super().__init__(config)
        self.config = config
        self.return_attn_weights = config.return_attn_weights

        self.bacformer = BacformerLargeModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ppi_head = BacformerProteinProteinInteractionHead(
            in_features=config.hidden_size,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        contig_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        return_attn_weights: bool = None,
        return_dict: bool | None = None,
        special_tokens_mask=None,  # for compatibility with Bacformer 26M
    ) -> OrderedDict | BacformerModelOutput:
        """Forward pass for protein-protein interaction prediction.

        Args:
            protein_embeddings: Pre-computed protein embeddings.
                Shape: (1, seq_len, hidden_dim). Batch size must be 1.
            contig_ids: Optional contig identifiers.
                Shape: (1, seq_len).
            attention_mask: Optional mask where 1 indicates valid tokens.
                Shape: (1, seq_len).
            labels: Tensor containing [protein1_idx, protein2_idx, interaction_label].
                Shape: (1, 3).
            return_attn_weights: Whether to return attention weights.
            return_dict: Whether to return a ModelOutput or tuple.
            special_tokens_mask: For compatibility with Bacformer 26M.

        Returns
        -------
            If return_dict=False: Tuple of (loss, logits).
            If return_dict=True: BacformerModelOutput with loss, logits,
                last_hidden_state, and optionally attentions.

        Raises
        ------
            AssertionError: If batch size is not 1.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        assert labels.shape[0] == 1, "Batch size should be 1 for protein-protein interaction task"

        outputs = self.bacformer(
            protein_embeddings=protein_embeddings,
            contig_ids=contig_ids,
            attention_mask=attention_mask,
            return_attn_weights=False,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state.squeeze(0)

        last_hidden_state = self.dense(self.dropout(last_hidden_state))
        last_hidden_state = torch.cat([last_hidden_state[labels[:, 0]], last_hidden_state[labels[:, 1]]], dim=0).mean(
            dim=0
        )
        logits = self.ppi_head(last_hidden_state)

        loss = F.binary_cross_entropy_with_logits(logits, labels[:, 2].type_as(logits).squeeze(0))

        if not return_dict:
            return (
                loss,
                logits,
            )

        return BacformerModelOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )
