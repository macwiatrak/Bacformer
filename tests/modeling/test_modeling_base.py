import torch
from bacformer.modeling.config import SPECIAL_TOKENS_DICT, BacformerConfig
from bacformer.modeling.modeling_base import BacformerEncoder, BacformerModel


def test_bacformer_encoder():
    batch_size = 2
    seq_len = 512
    dim = 480

    config = BacformerConfig(hidden_size=dim)
    model = BacformerEncoder(config)

    hidden_states = torch.randn(batch_size, seq_len, dim)
    last_hidden_states, attn_weights = model(hidden_states, return_attn_weights=True, is_causal=True)
    assert last_hidden_states.shape == (batch_size, seq_len, dim)
    assert isinstance(attn_weights, list)
    assert len(attn_weights) == config.num_hidden_layers
    assert attn_weights[0].shape == (batch_size, config.num_attention_heads, seq_len, seq_len)


def test_bacformer_model():
    batch_size = 2
    seq_len = 512
    dim = 480

    config = BacformerConfig(hidden_size=dim)
    model = BacformerModel(config, add_pooling_layer=False)

    protein_embeddings = torch.randn(batch_size, seq_len, dim)
    special_tokens_mask = torch.randint(0, SPECIAL_TOKENS_DICT["END"], (batch_size, seq_len))
    token_type_ids = torch.randint(0, config.max_token_type_embeddings, (batch_size, seq_len))
    output = model(
        protein_embeddings=protein_embeddings,
        special_tokens_mask=special_tokens_mask,
        token_type_ids=token_type_ids,
        return_dict=True,
        is_causal=True,
    )
    assert output.last_hidden_state.shape == (batch_size, seq_len, dim)
    assert not any(output.attentions)
    assert output.pooler_output is None

    model = BacformerModel(config, add_pooling_layer=True)
    output = model(
        protein_embeddings=protein_embeddings,
        special_tokens_mask=special_tokens_mask,
        token_type_ids=token_type_ids,
        return_dict=True,
        is_causal=True,
    )
    assert output.pooler_output.shape == (batch_size, dim)
