from typing import TypeAlias

from torch import Tensor

from .fp8 import FP8KVCache, quantize_fp8
from .int4 import Int4KVCache, quantize_int4

QuantizedKVCache: TypeAlias = FP8KVCache | Int4KVCache


def quantize_kv(
    k: Tensor,
    v: Tensor,
    format: str,
    granularity: str = "token",
    sink_tokens: int = 4,
) -> QuantizedKVCache:
    if format == "fp8_e4m3":
        return quantize_fp8(k, v, granularity, sink_tokens)
    if format == "int4":
        return quantize_int4(k, v, granularity, sink_tokens)
    raise ValueError("format must be fp8_e4m3 or int4")


def dense_storage_bytes(k: Tensor, v: Tensor) -> int:
    return k.nbytes + v.nbytes
