import pytest
import torch

from kvquant.fp8 import FP8_E4M3_MAX, quantize_fp8


def _tensors(sequence: int = 7, dim: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(13)
    k = torch.randn(2, 3, sequence, dim, generator=generator)
    v = torch.randn(2, 3, sequence, dim, generator=generator)
    k[:, :, :2] *= 20
    v[:, :, :2] *= 20
    return k, v


@pytest.mark.parametrize(
    ("granularity", "scale_shape"),
    [("head", (2, 3, 1, 1)), ("token", (2, 3, 5, 1))],
)
def test_fp8_granularity_and_exact_sinks(granularity: str, scale_shape: tuple[int, ...]) -> None:
    k, v = _tensors()
    cache = quantize_fp8(k, v, granularity=granularity, sink_tokens=2)
    k_restored, v_restored = cache.dequantize()
    assert cache.k_data.dtype == torch.float8_e4m3fn
    assert cache.k_scale.shape == scale_shape
    assert torch.equal(k_restored[:, :, :2], k[:, :, :2])
    assert torch.equal(v_restored[:, :, :2], v[:, :, :2])
    assert k_restored.shape == k.shape and v_restored.shape == v.shape


def test_fp8_uses_e4m3_range_and_finite_scales() -> None:
    k = torch.tensor([[[[0.0, 1.0, -2.0, 448.0]]]])
    cache = quantize_fp8(k, k, sink_tokens=0)
    restored, _ = cache.dequantize()
    assert FP8_E4M3_MAX == 448.0
    assert restored[0, 0, 0, 0] == 0
    assert restored[0, 0, 0, 3] == 448
    assert torch.isfinite(cache.k_scale).all()
    assert cache.k_data.float().abs().max() <= FP8_E4M3_MAX


def test_per_token_fp8_limits_outlier_token_damage() -> None:
    k = torch.full((1, 1, 8, 16), 0.01)
    k[:, :, -1] = 8.0
    head = quantize_fp8(k, k, "head", sink_tokens=0).dequantize()[0]
    token = quantize_fp8(k, k, "token", sink_tokens=0).dequantize()[0]
    head_error = (head[:, :, :-1] - k[:, :, :-1]).abs().mean()
    token_error = (token[:, :, :-1] - k[:, :, :-1]).abs().mean()
    assert token_error < head_error


def test_fp8_fp16_round_trip_restores_original_dtype() -> None:
    k, v = _tensors(dim=7)
    cache = quantize_fp8(k.half(), v.half(), sink_tokens=2)
    restored_k, restored_v = cache.dequantize()
    assert restored_k.dtype == torch.float16
    assert restored_v.dtype == torch.float16
