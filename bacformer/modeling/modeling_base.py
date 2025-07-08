import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn.functional import (
    scaled_dot_product_attention,
)
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from bacformer.modeling.config import BacformerConfig
from bacformer.modeling.utils import (
    create_4d_from_2d_attn_mask,
)


@dataclass
class BacformerModelOutput(ModelOutput):
    """Base class for outputs of the Bacformer model."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor | None = None
    attentions: list[torch.FloatTensor] | None = None
    pooler_output: torch.FloatTensor | None = None


# Taken from facebookresearch/llama/model.py
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape the rotary embeddings for broadcasting."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# Taken from facebookresearch/llama/model.py
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to the query and key tensors."""
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# Taken from facebookresearch/llama/model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the freqs cis for rotary embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore

    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def scaled_dot_product_attention_w_attn_weights(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch Native implementation, modified to return attention weights."""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    attn_output = attn_weight @ value
    return attn_output, attn_weight


class RotarySelfAttention(nn.Module):
    """Rotary self-attention module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_head = embed_dim // num_heads
        self.dropout_rate = dropout

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_proj_linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        is_causal: bool = False,
        return_attn_weights: bool = False,
    ):
        """Forward pass for the rotary self-attention module."""
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)
        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.num_heads, self.dim_head)
        xk = xk.view(batch_size, seq_len, self.num_heads, self.dim_head)
        xv = xv.view(batch_size, seq_len, self.num_heads, self.dim_head)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # Reshape for attention calculation: (b_sz, n_head, s_len, d_head)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        attn_weights = None
        if return_attn_weights:
            att, attn_weights = scaled_dot_product_attention_w_attn_weights(
                query=xq,
                key=xk,
                value=xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            att = scaled_dot_product_attention(
                query=xq,
                key=xk,
                value=xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=is_causal,
            )
        # Shape (b_sz, s_len, n_head, d_head)
        out = att.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.num_heads * self.dim_head)

        return self.att_proj_linear(out), attn_weights


class BacformerTransformerLayer(nn.Module):
    """Own implementation of transformer layer which uses pytorch native MHA but returns attention weights"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu"] = "gelu",
    ):
        super().__init__()
        self.self_mha = RotarySelfAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor = None,
        freqs_cos: torch.Tensor = None,
        freqs_sin: torch.Tensor = None,
        return_attn_weights: bool = False,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass"""
        attn_outputs, attn_weights = self.self_mha(
            hidden_state,
            attn_mask=attention_mask,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            return_attn_weights=return_attn_weights,
            is_causal=is_causal,
        )
        x = self.norm1(hidden_state + self.dropout1(attn_outputs))
        ff_output = self.fc2(self.dropout2(self.activation(self.fc1(x))))
        x = self.norm2(x + self.dropout3(ff_output))
        return x, attn_weights


class BacformerTransformerEncoder(nn.Module):
    """Own implementation of Transformer which return attention weights"""

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu"] = "gelu",
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                BacformerTransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor = None,
        freqs_cos: torch.Tensor = None,
        freqs_sin: torch.Tensor = None,
        return_attn_weights: bool = False,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Forward pass"""
        attn_weights_arr = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_state, attn_weights = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_state,
                    attention_mask,
                    freqs_cos,
                    freqs_sin,
                    return_attn_weights,
                    is_causal,
                )
            else:
                hidden_state, attn_weights = layer(
                    hidden_state=hidden_state,
                    attention_mask=attention_mask,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    return_attn_weights=return_attn_weights,
                    is_causal=is_causal,
                )
            # keep the attention weights from each layer
            attn_weights_arr.append(attn_weights)
        return hidden_state, attn_weights_arr


class BacformerEmbeddings(nn.Module):
    """Construct the protein embeddings from protein sequence, position embeddings and sequence type embeddings."""

    def __init__(self, config: BacformerConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.token_type_embeddings = nn.Embedding(
            num_embeddings=config.max_token_type_embeddings + 1,
            embedding_dim=config.hidden_size,
            padding_idx=config.max_token_type_embeddings,
        )

        self.special_tokens_embeddings = nn.Embedding(
            num_embeddings=config.num_special_tokens,
            embedding_dim=config.hidden_size,
        )
        self.prot_emb_token_id = config.prot_emb_token_id
        self.pad_token_id = config.pad_token_id

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        protein_embeddings: torch.Tensor = None,
        special_tokens_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,  # used for causal protein family modeling
        property_ids: torch.Tensor = None,  # used for conditional fine-tuning for desired property
    ) -> torch.Tensor:
        """Forward pass for protein embeddings."""
        bs, seq_length, dim = protein_embeddings.shape

        # pass the pooled ESM protein embeddings through a linear layer
        protein_embeddings = self.linear(protein_embeddings.type_as(self.linear.weight))
        protein_embeddings = torch.where(
            special_tokens_mask.unsqueeze(-1).repeat(1, 1, dim) == self.prot_emb_token_id,
            protein_embeddings,
            self.special_tokens_embeddings(special_tokens_mask),
        )

        if token_type_ids is not None:
            protein_embeddings += self.token_type_embeddings(token_type_ids)

        protein_embeddings = self.LayerNorm(protein_embeddings)
        protein_embeddings = self.dropout(protein_embeddings)
        return protein_embeddings


class BacformerProteinFamilyEmbeddings(nn.Module):
    """Construct the protein embeddings from protein family tokens, special tokens and sequence type embeddings."""

    def __init__(
        self,
        config: BacformerConfig,
        protein_family_embeddings: torch.Tensor = None,
        token_type_embeddings: torch.Tensor = None,
        special_tokens_embeddings: torch.Tensor = None,
        n_conditional_properties: int = None,
    ):
        super().__init__()
        self.config = config

        if protein_family_embeddings is not None:
            self.protein_family_embeddings = nn.Embedding.from_pretrained(
                protein_family_embeddings,
                freeze=False,
                padding_idx=config.pad_token_id,
            )
        else:
            self.protein_family_embeddings = nn.Embedding(
                num_embeddings=config.protein_clusters_vocab_size + 1,
                embedding_dim=config.hidden_size,
                padding_idx=config.pad_token_id,
            )

        if token_type_embeddings is not None:
            self.token_type_embeddings = nn.Embedding.from_pretrained(
                token_type_embeddings,
                freeze=False,
                padding_idx=config.max_token_type_embeddings,
            )
        else:
            self.token_type_embeddings = nn.Embedding(
                num_embeddings=config.max_token_type_embeddings + 1,
                embedding_dim=config.hidden_size,
                padding_idx=config.max_token_type_embeddings,
            )

        if special_tokens_embeddings is not None:
            self.special_tokens_embeddings = nn.Embedding.from_pretrained(
                special_tokens_embeddings,
                freeze=False,
                padding_idx=config.pad_token_id,
            )
        else:
            self.special_tokens_embeddings = nn.Embedding(
                num_embeddings=config.num_special_tokens,
                embedding_dim=config.hidden_size,
                padding_idx=config.pad_token_id,
            )

        # add layer for conditional properties
        if n_conditional_properties is not None:
            self.conditional_properties_layer = nn.Embedding(n_conditional_properties, config.hidden_size)

        self.prot_emb_token_id = config.prot_emb_token_id
        self.pad_token_id = config.pad_token_id

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        protein_embeddings: torch.Tensor = None,
        special_tokens_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,  # used for causal protein family modeling
        property_ids: torch.Tensor = None,  # used for conditional fine-tuning for desired property
    ) -> torch.Tensor:
        """Forward pass for protein embeddings."""
        # pass the pooled ESM protein embeddings through a linear layer
        # replace -100 with pad_token_id
        labels[labels == -100] = self.pad_token_id
        protein_embeddings = self.protein_family_embeddings(labels)

        bs, seq_length, dim = protein_embeddings.shape
        protein_embeddings = torch.where(
            special_tokens_mask.unsqueeze(-1).repeat(1, 1, dim) == self.prot_emb_token_id,
            protein_embeddings,
            self.special_tokens_embeddings(special_tokens_mask),
        )

        if token_type_ids is not None:
            protein_embeddings += self.token_type_embeddings(token_type_ids)

        if property_ids is not None:
            # get the embeddings for the conditional properties
            property_embedding = self.conditional_properties_layer(property_ids).unsqueeze(1)
            # concatenate the protein embeddings with the conditional properties embeddings
            # property embeddings are added to the beginning of the protein embeddings after the CLS token
            protein_embeddings = torch.cat(
                [
                    protein_embeddings[:, :1, :],  # CLS token
                    property_embedding,  # conditional properties embeddings
                    protein_embeddings[:, 1:, :],
                ],  # protein embeddings
                dim=1,
            )

        protein_embeddings = self.LayerNorm(protein_embeddings)
        protein_embeddings = self.dropout(protein_embeddings)
        return protein_embeddings


class BacformerEncoder(nn.Module):
    """Bacformer encoder model"""

    def __init__(self, config: BacformerConfig):
        super().__init__()
        self.config = config

        self.encoder = BacformerTransformerEncoder(
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            activation="gelu",
            dropout=config.attention_probs_dropout_prob,
        )

        # Note that config.max_position_embeddings is multiplied by 1.5 because the token limit for the Bacformer of
        # models is 6000. Adding this multiplier instead of using 6000 directly allows for dynamism of token
        # lengths while training or fine-tuning.
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, int(config.max_position_embeddings * 1.5)
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_attn_weights: bool | None = None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Pass the input through the encoder layers in turn.

        Args:
            hidden_states: hidden states from the BacformerEmbeddings layer
            attention_mask: mask for the attention in the transformer
        """
        return_attn_weights = (
            return_attn_weights if return_attn_weights is not None else self.config.return_attn_weights
        )
        bs, seq_len, _ = hidden_states.shape
        last_hidden_state, attn_weights = self.encoder(
            hidden_state=hidden_states,
            attention_mask=attention_mask,
            freqs_cos=self.freqs_cos[:seq_len, :],
            freqs_sin=self.freqs_sin[:seq_len, :],
            return_attn_weights=return_attn_weights,
            is_causal=is_causal,
        )
        return last_hidden_state, attn_weights


# Copied from transformers.models.bert.modeling_bert.BertPooler
class BacformerPooler(nn.Module):
    """Pooler for Bacformer model."""

    def __init__(self, config: BacformerConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward method for the pooler."""
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


class BacformerPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models."""

    config_class = BacformerConfig
    base_model_prefix = "bacformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BacformerEmbeddings", "BacformerTransformerLayer"]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class BacformerModel(BacformerPreTrainedModel):
    """Bacformer model."""

    def __init__(self, config: BacformerConfig, add_pooling_layer: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = BacformerEmbeddings(config)
        self.encoder = BacformerEncoder(config)

        self.pooler = BacformerPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        protein_embeddings: torch.Tensor = None,
        special_tokens_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,  # used only for protein family generation
        property_ids: torch.Tensor = None,  # used only for protein family generation
        return_attn_weights: bool = False,
        return_dict: bool | None = None,
        is_causal: bool = False,
    ) -> BacformerModelOutput | None:
        """Forward method for the model."""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        # get embeddings
        protein_embeddings = self.embeddings(
            protein_embeddings=protein_embeddings,
            labels=labels,
            special_tokens_mask=special_tokens_mask,
            token_type_ids=token_type_ids,
            property_ids=property_ids,
        )

        # create 3D attention mask from 2D if not doing causal GM
        if attention_mask is not None and not is_causal:
            attention_mask = create_4d_from_2d_attn_mask(
                attn_mask=attention_mask, num_attn_heads=self.config.num_attention_heads
            ).bool()

        last_hidden_state, attentions = self.encoder(
            hidden_states=protein_embeddings,
            attention_mask=attention_mask,
            return_attn_weights=return_attn_weights,
            is_causal=is_causal,
        )
        pooler_output = (
            self.pooler(hidden_states=last_hidden_state, padding_mask=attention_mask)
            if self.pooler is not None
            else None
        )

        if not return_dict:
            return (last_hidden_state, pooler_output, attentions)

        return BacformerModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            attentions=attentions,
        )
