import torch
from bacformer.modeling.config import BacformerConfig
from bacformer.modeling.modeling_base import BacformerModel
from bacformer.modeling.modeling_large import BacformerLargeConfig, BacformerLargeModel


def _base_config():
    return BacformerConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        hidden_size=8,
        intermediate_size=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=8,
        max_token_type_embeddings=4,
        protein_clusters_vocab_size=8,
    )


def _large_config():
    return BacformerLargeConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=8,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=8,
        max_token_type_embeddings=4,
        protein_clusters_vocab_size=8,
    )


def _expected_base_buffers(config):
    dim = config.hidden_size // config.num_attention_heads
    end = int(config.max_position_embeddings * 1.5)
    inv_freq = 1 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(end, dtype=torch.float32), inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def _expected_large_buffer(config):
    dim = config.hidden_size // config.num_attention_heads
    return 1 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def test_base_init_hook_reconstructs_rotary_buffers():
    with torch.device("meta"):
        model = BacformerModel(_base_config())
    model.to_empty(device="cpu")
    model.encoder.freqs_cos.zero_()
    model.encoder.freqs_sin.zero_()

    model._init_weights(model.encoder)

    expected_cos, expected_sin = _expected_base_buffers(model.config)
    torch.testing.assert_close(model.encoder.freqs_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(model.encoder.freqs_sin, expected_sin, rtol=0, atol=0)
    assert not any("freqs_cos" in key or "freqs_sin" in key for key in model.state_dict())


def test_large_init_hook_reconstructs_all_rotary_buffers():
    with torch.device("meta"):
        model = BacformerLargeModel(_large_config())
    model.to_empty(device="cpu")
    expected = _expected_large_buffer(model.config)

    for layer in model.encoder.layers:
        layer.attn.rotary.inv_freq.zero_()
        model._init_weights(layer.attn.rotary)

    for layer in model.encoder.layers:
        torch.testing.assert_close(layer.attn.rotary.inv_freq, expected, rtol=0, atol=0)
    assert not any("inv_freq" in key for key in model.state_dict())


def test_base_rotary_buffers_are_reconstructed_after_pretrained_round_trip(tmp_path):
    model = BacformerModel(_base_config()).eval()
    expected_cos, expected_sin = _expected_base_buffers(model.config)
    torch.testing.assert_close(model.encoder.freqs_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(model.encoder.freqs_sin, expected_sin, rtol=0, atol=0)

    model.save_pretrained(tmp_path)
    loaded = BacformerModel.from_pretrained(tmp_path).eval()

    torch.testing.assert_close(loaded.encoder.freqs_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(loaded.encoder.freqs_sin, expected_sin, rtol=0, atol=0)


def test_large_rotary_buffers_are_reconstructed_after_pretrained_round_trip(tmp_path):
    model = BacformerLargeModel(_large_config()).eval()
    expected = _expected_large_buffer(model.config)
    for layer in model.encoder.layers:
        torch.testing.assert_close(layer.attn.rotary.inv_freq, expected, rtol=0, atol=0)

    model.save_pretrained(tmp_path)
    loaded = BacformerLargeModel.from_pretrained(tmp_path).eval()

    for layer in loaded.encoder.layers:
        torch.testing.assert_close(layer.attn.rotary.inv_freq, expected, rtol=0, atol=0)
