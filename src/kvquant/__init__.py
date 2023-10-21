"""Quantized KV caches with full-precision attention sinks."""

from .cache import QuantizedKVCache, dense_storage_bytes, quantize_kv
from .fp8 import FP8KVCache, quantize_fp8
from .int4 import Int4KVCache, quantize_int4

__version__ = "0.1.0"

__all__ = [
    "FP8KVCache",
    "Int4KVCache",
    "QuantizedKVCache",
    "dense_storage_bytes",
    "quantize_fp8",
    "quantize_int4",
    "quantize_kv",
]
