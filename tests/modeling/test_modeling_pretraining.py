import torch
from bacformer.modeling.config import SPECIAL_TOKENS_DICT, BacformerConfig
from bacformer.modeling.modeling_pretraining import (
    BacformerForCausalGM,
    BacformerForCausalProteinFamilyModeling,
    BacformerForMaskedGM,
    BacformerForMaskedGMWithContrastiveLoss,
)


def test_bacfomer_for_causal_gm():
    batch_size = 2
    seq_len = 512
    dim = 480

    config = BacformerConfig(hidden_size=dim)
    model = BacformerForCausalGM(config)

    protein_embeddings = torch.randn(batch_size, seq_len, dim)
    special_tokens_mask = torch.randint(0, SPECIAL_TOKENS_DICT["END"], (batch_size, seq_len))
    token_type_ids = torch.randint(0, config.max_token_type_embeddings, (batch_size, seq_len))
    labels = torch.randint(0, config.protein_clusters_vocab_size, (batch_size, seq_len))
    output = model(
        protein_embeddings=protein_embeddings,
        special_tokens_mask=special_tokens_mask,
        token_type_ids=token_type_ids,
        labels=labels,
        return_dict=True,
    )
    assert output.last_hidden_state.shape == (batch_size, seq_len, dim)
    assert output.loss > 0
    assert not any(output.attentions)
    assert output.logits.shape == (batch_size, seq_len, config.protein_clusters_vocab_size + 1)


def test_bacformer_for_masked_gm():
    batch_size = 2
    seq_len = 512
    dim = 480

    config = BacformerConfig(hidden_size=dim)
    model = BacformerForMaskedGM(config)

    protein_embeddings = torch.randn(batch_size, seq_len, dim)
    special_tokens_mask = torch.randint(0, SPECIAL_TOKENS_DICT["END"], (batch_size, seq_len))
    token_type_ids = torch.randint(0, config.max_token_type_embeddings, (batch_size, seq_len))
    attention_mask = torch.randint(0, 2, (batch_size, seq_len))
    labels = torch.randint(0, config.protein_clusters_vocab_size, (batch_size, seq_len))
    output = model(
        protein_embeddings=protein_embeddings,
        special_tokens_mask=special_tokens_mask,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )
    assert output.last_hidden_state.shape == (batch_size, seq_len, dim)
    assert output.loss > 0
    assert not any(output.attentions)
    assert output.logits.shape == (batch_size * seq_len, config.protein_clusters_vocab_size + 1)


def test_bacformer_for_causal_protein_family_modeling():
    batch_size = 2
    seq_len = 512
    dim = 480
    n_conditional_properties = 5

    config = BacformerConfig(hidden_size=dim)
    model = BacformerForCausalProteinFamilyModeling(config, n_conditional_properties=n_conditional_properties)

    special_tokens_mask = torch.randint(0, SPECIAL_TOKENS_DICT["END"], (batch_size, seq_len))
    token_type_ids = torch.randint(0, config.max_token_type_embeddings, (batch_size, seq_len))
    labels = torch.randint(0, config.protein_clusters_vocab_size, (batch_size, seq_len))
    property_ids = torch.randint(0, n_conditional_properties, (batch_size,))

    output = model(
        special_tokens_mask=special_tokens_mask,
        token_type_ids=token_type_ids,
        labels=labels,
        property_ids=property_ids,
        return_dict=True,
    )
    assert output.last_hidden_state.shape == (batch_size, seq_len + 1, dim)
    assert output.loss > 0
    assert not any(output.attentions)
    assert output.logits.shape == (batch_size, seq_len + 1, config.protein_clusters_vocab_size + 1)


def test_bacformer_for_masked_gm_with_contrastive_loss():
    batch_size = 1
    seq_len = 512
    dim = 480

    config = BacformerConfig(hidden_size=dim)
    model = BacformerForMaskedGMWithContrastiveLoss(config)

    protein_embeddings = torch.randn(batch_size, seq_len, dim)
    special_tokens_mask = torch.randint(0, SPECIAL_TOKENS_DICT["END"], (batch_size, seq_len))
    token_type_ids = torch.randint(0, config.max_token_type_embeddings, (batch_size, seq_len))
    attention_mask = torch.randint(0, 2, (batch_size, seq_len))
    labels = torch.randint(0, config.protein_clusters_vocab_size, (batch_size, seq_len))
    output = model(
        protein_embeddings=protein_embeddings,
        special_tokens_mask=special_tokens_mask,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )
    assert output.last_hidden_state.shape == (batch_size, seq_len, dim)
    assert output.loss > 0
    assert not any(output.attentions)
    assert output.logits.shape == (batch_size * seq_len, config.protein_clusters_vocab_size + 1)
