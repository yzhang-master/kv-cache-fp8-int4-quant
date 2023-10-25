import pytest
import torch

from kvquant.cache import quantize_kv


@pytest.mark.parametrize("format", ["fp8_e4m3", "int4"])
@pytest.mark.parametrize("granularity", ["head", "token"])
def test_all_sink_fp8_and_int4_round_trip_exactly(format: str, granularity: str) -> None:
    k = torch.randn(1, 2, 3, 5, generator=torch.Generator().manual_seed(2))
    cache = quantize_kv(k, k, format, granularity, sink_tokens=3)
    restored_k, restored_v = cache.dequantize()
    assert cache.k_data.shape[2] == 0
    if granularity == "token":
        assert cache.storage_bytes == k.nbytes * 2
    assert torch.equal(restored_k, k)
    assert torch.equal(restored_v, k)


def test_invalid_format_is_rejected() -> None:
    k = torch.randn(1, 1, 2, 4)
    with pytest.raises(ValueError, match="format"):
        quantize_kv(k, k, "int8")
