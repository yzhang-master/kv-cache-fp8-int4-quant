import pytest
import torch

from kvquant.attention import (
    attention_metrics,
    causal_attention,
    make_long_context_proxy,
    quantized_attention,
)
from kvquant.cache import quantize_kv


def test_causal_attention_masks_future_and_normalizes() -> None:
    generator = torch.Generator().manual_seed(31)
    q = torch.randn(1, 2, 16, 8, generator=generator)
    k = torch.randn(1, 2, 16, 8, generator=generator)
    v = torch.randn(1, 2, 16, 8, generator=generator)
    output, probabilities = causal_attention(q, k, v)
    assert output.shape == q.shape
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones_like(output[..., 0]))
    assert torch.count_nonzero(torch.triu(probabilities, diagonal=1)) == 0
    assert torch.equal(probabilities[0, 0, 0, 1:], torch.zeros(15))


@pytest.mark.parametrize("format", ["fp8_e4m3", "int4"])
def test_real_attention_on_quantized_long_context_proxy(format: str) -> None:
    q, k, v = make_long_context_proxy(sequence_length=96, heads=2, head_dim=16)
    reference, reference_probabilities = causal_attention(q, k, v)
    cache = quantize_kv(k, v, format, "token", sink_tokens=4)
    candidate, candidate_probabilities = quantized_attention(q, cache)
    metrics = attention_metrics(
        reference, reference_probabilities, candidate, candidate_probabilities, sink_tokens=4
    )
    assert candidate.shape == reference.shape
    assert 0 <= metrics["attention_top1_agreement"] <= 1
    assert metrics["mean_output_cosine"] > 0.9
    assert metrics["relative_output_error"] < 0.25


def test_attention_metrics_detect_changed_output() -> None:
    q, k, v = make_long_context_proxy(sequence_length=16, heads=1, head_dim=8)
    output, probabilities = causal_attention(q, k, v)
    perfect = attention_metrics(output, probabilities, output, probabilities, 4)
    changed = attention_metrics(output, probabilities, output + 1, probabilities.roll(1, -1), 4)
    assert perfect["relative_output_error"] == 0
    assert perfect["attention_top1_agreement"] == 1
    assert changed["relative_output_error"] > 0
    assert changed["attention_top1_agreement"] < 1
