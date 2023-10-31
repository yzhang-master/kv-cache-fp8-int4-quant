import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from .attention import (
    attention_metrics,
    causal_attention,
    make_long_context_proxy,
    quantized_attention,
)
from .cache import dense_storage_bytes, quantize_kv


def _measure_attention(function, repeats: int) -> float:
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        durations.append((time.perf_counter() - start) * 1000)
    return statistics.median(durations)


def run_benchmark(
    output: Path,
    sequence_length: int = 1024,
    heads: int = 4,
    head_dim: int = 32,
    sink_tokens: int = 4,
    repeats: int = 3,
    threads: int = 1,
) -> dict:
    torch.set_num_threads(threads)
    q, k, v = make_long_context_proxy(sequence_length, heads, head_dim, sink_tokens)
    reference, reference_probabilities = causal_attention(q, k, v)
    positions = torch.arange(sequence_length)
    targets = (positions * 37) % (positions + 1)
    target_rate = (reference_probabilities.argmax(dim=-1) == targets).float().mean().item()
    dense_fp32 = dense_storage_bytes(k, v)
    dense_fp16 = dense_fp32 // 2
    rows = [
        {
            "format": "fp32",
            "granularity": "dense",
            "sink_tokens": sink_tokens,
            "relative_output_error": 0.0,
            "mean_output_cosine": 1.0,
            "attention_top1_agreement": 1.0,
            "mean_absolute_sink_mass_change": 0.0,
            "storage_bytes": dense_fp32,
            "saved_vs_fp32_percent": 0.0,
            "saved_vs_fp16_percent": 100 * (1 - dense_fp32 / dense_fp16),
            "attention_ms": _measure_attention(lambda: causal_attention(q, k, v), repeats),
        }
    ]
    fp16_k, fp16_v = k.half(), v.half()
    fp16_output, fp16_probabilities = causal_attention(q, fp16_k, fp16_v)
    rows.append(
        {
            "format": "fp16",
            "granularity": "dense",
            "sink_tokens": sink_tokens,
            **attention_metrics(
                reference, reference_probabilities, fp16_output, fp16_probabilities, sink_tokens
            ),
            "storage_bytes": dense_fp16,
            "saved_vs_fp32_percent": 50.0,
            "saved_vs_fp16_percent": 0.0,
            "attention_ms": _measure_attention(
                lambda: causal_attention(q, fp16_k, fp16_v), repeats
            ),
        }
    )
    for format in ("fp8_e4m3", "int4"):
        for granularity in ("token", "head"):
            for retained_sinks in (0, sink_tokens):
                start = time.perf_counter()
                cache = quantize_kv(k, v, format, granularity, retained_sinks)
                quantization_ms = (time.perf_counter() - start) * 1000
                candidate, candidate_probabilities = quantized_attention(q, cache)
                storage = cache.storage_bytes
                rows.append(
                    {
                        "format": format,
                        "granularity": granularity,
                        "sink_tokens": retained_sinks,
                        **attention_metrics(
                            reference,
                            reference_probabilities,
                            candidate,
                            candidate_probabilities,
                            retained_sinks,
                        ),
                        "storage_bytes": storage,
                        "saved_vs_fp32_percent": 100 * (1 - storage / dense_fp32),
                        "saved_vs_fp16_percent": 100 * (1 - storage / dense_fp16),
                        "quantization_ms": quantization_ms,
                        "attention_ms": _measure_attention(
                            lambda cache=cache: quantized_attention(q, cache), repeats
                        ),
                    }
                )
    result = {
        "scope": (
            "Synthetic long-range causal-attention proxy on CPU; "
            "real K/V quantization and attention; "
            "FP8 uses torch.float8_e4m3fn storage conversion, not hardware FP8 execution"
        ),
        "config": {
            "sequence_length": sequence_length,
            "heads": heads,
            "head_dim": head_dim,
            "batch_size": 1,
            "maximum_sink_tokens": sink_tokens,
            "repeats": repeats,
            "threads": threads,
            "seed": 71,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "reference_long_range_top1_rate": target_rate,
        "storage_note": (
            "K and V payload, fp32 scales, uint8 zero points, and original-precision sink tensors; "
            "no allocator overhead"
        ),
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/long_context_cpu.json"))
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    options = vars(args).copy()
    options.pop("smoke")
    if args.smoke:
        options.update(sequence_length=64, heads=2, head_dim=16, repeats=1)
    result = run_benchmark(**options)
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
