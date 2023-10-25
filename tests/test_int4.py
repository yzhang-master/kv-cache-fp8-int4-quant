import pytest
import torch

from kvquant.cache import dense_storage_bytes, quantize_kv
from kvquant.int4 import _pack, _unpack, quantize_int4


def test_int4_pack_low_nibble_first_and_odd_width() -> None:
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0]]]])
    packed = _pack(q)
    assert packed.tolist() == [[[[0x21, 0x43, 0x05]]]]
    assert _unpack(packed, 5).tolist() == q.tolist()


@pytest.mark.parametrize(
    ("granularity", "scale_shape"),
    [("head", (1, 2, 1, 1)), ("token", (1, 2, 5, 1))],
)
def test_int4_granularity_sinks_and_odd_dim(granularity: str, scale_shape: tuple[int, ...]) -> None:
    generator = torch.Generator().manual_seed(21)
    k = torch.randn(1, 2, 8, 7, generator=generator)
    v = torch.randn(1, 2, 8, 7, generator=generator)
    cache = quantize_int4(k, v, granularity, sink_tokens=3)
    restored_k, restored_v = cache.dequantize()
    assert cache.k_data.dtype == torch.uint8
    assert cache.k_data.shape == (1, 2, 5, 4)
    assert cache.k_scale.shape == scale_shape
    assert torch.equal(restored_k[:, :, :3], k[:, :, :3])
    assert torch.equal(restored_v[:, :, :3], v[:, :, :3])
    assert cache.k_data.shape[-1] * 2 - 1 == k.shape[-1]


def test_int4_constant_negative_and_zero_tokens() -> None:
    k = torch.tensor([[[[0.0, 0.0], [-3.0, -3.0], [2.0, 2.0]]]])
    cache = quantize_int4(k, k, sink_tokens=0)
    restored, _ = cache.dequantize()
    assert torch.equal(restored, k)


def test_per_token_int4_protects_normal_tokens_from_outlier() -> None:
    k = torch.full((1, 1, 8, 16), 0.01)
    k[:, :, -1] = 8.0
    head = quantize_int4(k, k, "head", sink_tokens=0).dequantize()[0]
    token = quantize_int4(k, k, "token", sink_tokens=0).dequantize()[0]
    head_error = (head[:, :, :-1] - k[:, :, :-1]).abs().mean()
    token_error = (token[:, :, :-1] - k[:, :, :-1]).abs().mean()
    assert token_error < head_error * 0.1


def test_storage_counts_real_payload_and_dense_baseline() -> None:
    k = torch.randn(1, 2, 8, 16)
    cache = quantize_int4(k, k, "token", sink_tokens=2)
    expected = 2 * (1 * 2 * 6 * 8) + 2 * (1 * 2 * 6 * 1) * 5 + 2 * (1 * 2 * 2 * 16) * 4
    assert cache.storage_bytes == expected
    assert dense_storage_bytes(k, k) == 2 * k.numel() * 4
    assert cache.storage_bytes < dense_storage_bytes(k, k)


def test_all_sink_cache_needs_no_quantized_values() -> None:
    k = torch.randn(1, 1, 3, 4)
    cache = quantize_kv(k, k, "int4", sink_tokens=3)
    restored, _ = cache.dequantize()
    assert cache.k_data.shape[2] == 0
    assert torch.equal(restored, k)
