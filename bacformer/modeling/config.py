from typing import Literal

from transformers import PretrainedConfig

SPECIAL_TOKENS_DICT = {
    "PAD": 0,
    "MASK": 1,
    "CLS": 2,
    "SEP": 3,
    "PROT_EMB": 4,
    "END": 5,
}


class BacformerConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `BacformerModel`."""

    model_type = "bacformer"

    def __init__(
        self,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        hidden_size: int = 480,  # default esm2_t6_8M_UR50D embedding dim
        intermediate_size: int = 1280,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 6000,
        max_token_type_embeddings: int = 1000,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        pad_token_id: int = SPECIAL_TOKENS_DICT["PAD"],
        mask_token_id: int = SPECIAL_TOKENS_DICT["MASK"],
        prot_emb_token_id: int = SPECIAL_TOKENS_DICT["PROT_EMB"],
        end_token_id: int = SPECIAL_TOKENS_DICT["END"],
        num_special_tokens: int = len(SPECIAL_TOKENS_DICT),
        protein_clusters_vocab_size: int = 50000,  # equal to the nr of protein clusters
        num_labels: int = 1,  # for downstream tasks
        is_causal_gm: bool = False,
        return_dict: bool = False,
        return_attn_weights: bool = False,
        alpha_contrastive_loss: float = 0.5,
        # only to use in the BacformerForGenomeClassification
        problem_type: Literal[
            "regression", "binary_classification", "single_label_classification", "multi_label_classification"
        ] = "single_label_classification",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_token_type_embeddings = max_token_type_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.prot_emb_token_id = prot_emb_token_id
        self.end_token_id = end_token_id
        self.num_special_tokens = num_special_tokens
        self.protein_clusters_vocab_size = protein_clusters_vocab_size
        self.num_labels = num_labels
        self.is_causal_gm = is_causal_gm
        self.return_dict = return_dict
        self.return_attn_weights = return_attn_weights
        self.problem_type = problem_type
        self.alpha_contrastive_loss = alpha_contrastive_loss
