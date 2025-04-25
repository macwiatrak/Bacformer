import torch
from bacformer.modeling.config import SPECIAL_TOKENS_DICT, BacformerConfig
from bacformer.modeling.modeling_tasks import (
    BacformerForGenomeClassification,
    BacformerForProteinClassification,
    BacformerForProteinProteinInteraction,
)


def test_bacformer_for_protein_classification():
    batch_size = 2
    seq_len = 512
    dim = 480
    n_labels = 10

    config = BacformerConfig(
        hidden_size=dim,
        problem_type="single_label_classification",
        num_labels=n_labels,
    )
    model = BacformerForProteinClassification(config)

    protein_embeddings = torch.randn(batch_size, seq_len, dim)
    special_tokens_mask = torch.randint(0, SPECIAL_TOKENS_DICT["END"], (batch_size, seq_len))
    token_type_ids = torch.randint(0, config.max_token_type_embeddings, (batch_size, seq_len))
    attention_mask = torch.randint(0, 2, (batch_size, seq_len))
    labels = torch.randint(0, n_labels, (batch_size, seq_len))
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
    assert output.logits.shape == (batch_size, seq_len, n_labels)


def test_bacformer_for_genome_classification():
    batch_size = 2
    seq_len = 512
    dim = 480
    n_labels = 10

    config = BacformerConfig(
        hidden_size=dim,
        num_labels=n_labels,
    )
    model = BacformerForGenomeClassification(config)

    protein_embeddings = torch.randn(batch_size, seq_len, dim)
    special_tokens_mask = torch.randint(0, SPECIAL_TOKENS_DICT["END"], (batch_size, seq_len))
    token_type_ids = torch.randint(0, config.max_token_type_embeddings, (batch_size, seq_len))
    attention_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.float32)
    labels = torch.randint(0, n_labels, (batch_size,))
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
    assert output.logits.shape == (batch_size, n_labels)


def test_bacformer_for_protein_protein_interaction():
    batch_size = 1
    seq_len = 512
    dim = 480
    n_pairs = 100

    config = BacformerConfig(hidden_size=dim)
    model = BacformerForProteinProteinInteraction(config)

    protein_embeddings = torch.randn(batch_size, seq_len, dim)
    special_tokens_mask = torch.randint(0, SPECIAL_TOKENS_DICT["END"], (batch_size, seq_len))
    token_type_ids = torch.randint(0, config.max_token_type_embeddings, (batch_size, seq_len))
    attention_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.float32)

    ppi_pairs = torch.randint(0, seq_len - 3, (n_pairs, 2))
    labels = torch.randint(0, 2, (n_pairs,))
    labels = torch.cat([ppi_pairs, labels.unsqueeze(1)], dim=1)
    output = model(
        protein_embeddings=protein_embeddings,
        special_tokens_mask=special_tokens_mask,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )
    assert output.last_hidden_state.shape == (n_pairs, dim)
    assert output.loss > 0
    assert not any(output.attentions)
    assert output.logits.shape == (n_pairs,)
