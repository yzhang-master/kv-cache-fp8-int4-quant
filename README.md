# kv-cache-fp8-int4-quant

## Context

**Techniques from 2023–2024, reimplemented in 2026.** Era 2: Senior AI Engineer, long-context inference. This repository implements per-head and per-token FP8-E4M3 and INT4 K/V caches, retains initial attention sinks in their original precision, and evaluates real causal attention on a 1,024-token CPU proxy.

## Problem

A transformer KV cache grows linearly with sequence length. Lower-precision K/V tensors reduce storage and potential bandwidth, but one large outlier token can dominate a head-wide scale. Initial attention-sink tokens can also have disproportionate influence; quantizing them may damage attention even when average reconstruction error looks acceptable. The evaluation therefore needs real attention outputs and attention decisions, not just compressed tensor sizes.

## Method

Inputs have shape `[batch, heads, sequence, head_dim]`, with matching K and V tensors. The first `sink_tokens` vectors are cloned without quantization. Only non-sink vectors enter the quantized payload; dequantization concatenates the original-precision prefix with the restored suffix.

**Per-token scales** reduce over `head_dim`, producing one scale per `[batch, head, token]`. **Per-head scales** reduce over both sequence and head dimensions, producing one scale per `[batch, head]`.

For FP8, symmetric scaling is

```
scale = max(abs(x)) / 448
q = float8_e4m3fn(clamp(x / scale, -448, 448))
x_hat = float32(q) * scale
```

Zero ranges use scale one. Storage uses torch's explicit `torch.float8_e4m3fn` conversion, including its E4M3 rounding and saturation behavior. All arithmetic after storage is fp32. **No hardware FP8 support or FP8 attention execution is claimed.**

For INT4, each scale group uses asymmetric unsigned quantization:

```
scale = (max(x) - min(x)) / 15
zero = clamp(round(-min(x) / scale), 0, 15)
q = clamp(round(x / scale) + zero, 0, 15)
x_hat = (q - zero) * scale
```

Two 4-bit values are packed into every `uint8`, low nibble first. Odd head dimensions have one padding nibble. Zero and nonzero constant groups are handled explicitly. Dequantization unpacks actual nibbles before fp32 attention.

The attention proxy constructs queries from long-range earlier keys and amplifies initial sink K/V tensors. It computes full causal scores, softmax probabilities, and value outputs. Metrics are relative output Frobenius error, mean output cosine, agreement of each row's most-attended key, and absolute sink-attention-mass change. Storage counts both K and V payloads, fp32 scales, uint8 zero points, and original-precision sinks; allocator overhead is excluded.

## Results

Measured with Python 3.12.3 and torch 2.15.0+cpu, one CPU thread, batch 1, four heads, head dimension 32, sequence length 1,024, seed 71, and three timing repetitions. Raw values, including quantization and attention timings, are in `results/long_context_cpu.json`.

With four full-precision attention sinks:

| format | scale granularity | relative output error | top-1 agreement | cache bytes | saved vs fp32 |
|---|---|---:|---:|---:|---:|
| fp32 | dense | 0 | 1.000000 | 1,048,576 | 0% |
| fp16 | dense | 0.000602 | 0.999756 | 524,288 | 50.00% |
| FP8-E4M3 | token | **0.008937** | **0.999023** | 297,856 | 71.59% |
| FP8-E4M3 | head | 0.011315 | 0.997070 | 265,248 | 74.70% |
| INT4 | token | 0.022945 | 0.995850 | 175,456 | 83.27% |
| INT4 | head | 0.047369 | 0.992676 | **134,696** | **87.15%** |

Retaining four sinks substantially improved this proxy:

| format / granularity | error, no retained sinks | error, four sinks | error reduction |
|---|---:|---:|---:|
| FP8 / token | 0.065397 | 0.008937 | 86.33% |
| FP8 / head | 0.071236 | 0.011315 | 84.12% |
| INT4 / token | 0.210355 | 0.022945 | 89.09% |
| INT4 / head | 0.375806 | 0.047369 | 87.40% |

Per-token INT4 is more accurate than per-head INT4 but needs more scales. FP8 per-token has the lowest quantized output error; INT4 per-head has the smallest storage. These are proxy tradeoffs, not a universal accuracy ranking.

The full-precision query generator's intended key is the attention argmax for **32.9102%** of rows. Thus `top-1 agreement` measures preservation of the full-precision attention decision, **not** absolute retrieval-task accuracy. CPU attention timings in the JSON include dequantization and show no consistent quantized speedup at this scale.

**GPU long-context metrics: pending.** No GPU is available on this machine, so no GPU number is reported. On a CUDA host with a torch build supporting `float8_e4m3fn` conversion, run the real benchmark path:

```bash
PYTHON_BIN=python scripts/benchmark.sh --device cuda \
  --sequence-length 8192 --heads 8 --head-dim 64 --sink-tokens 4 \
  --repeats 20 --output results/long_context_cuda.json
```

This command materializes full attention matrices and needs several GiB of memory. CUDA events are not required because the harness synchronizes around each timed region. It still performs fp32 attention after dequantization; it is not a fused CUDA FP8/INT4 attention kernel. The CUDA path is implemented but unverified here.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check .
pytest -q
scripts/smoke.sh
scripts/benchmark.sh --smoke --output results/smoke.tmp
scripts/benchmark.sh --output results/long_context_cpu.json
```

On this workspace, use `PYTHON_BIN=/home/pus/.venvs/portfolio/bin/python`. The CPU benchmark is deterministic in accuracy metrics; latency values vary by host and run. No model download, transformers package, network access, or GPU is needed.

The 23-test suite covers FP8 range and dtype, per-head/per-token scales, packed nibble order, odd dimensions, constant groups, outlier isolation, exact sinks (including all-sink caches), storage accounting, causal masks, real attention metrics, and benchmark output. Mutation checks removed FP8 dequantization scales, INT4 zero points, and full-precision sink storage; each made pytest fail. Source was restored and all 23 tests passed again.

## Limitations

- The long-context input is synthetic attention, not a pretrained transformer, natural-language long-context benchmark, perplexity evaluation, or incremental decoding workload.
- Full `[sequence, sequence]` attention probabilities are materialized for metrics. Production flash/paged attention would use less temporary memory and different kernels.
- FP8 support is storage conversion through an explicit CPU-supported torch dtype. It is neither a custom emulation claim nor evidence of hardware FP8 acceleration.
- Quantization is offline over complete K/V tensors. A serving cache needs incremental append logic, block/page integration, and scale-management policy.
- Storage is analytical tensor byte accounting, not process RSS or allocator peak memory. Query tensors, activations, and attention workspaces are excluded.
- GPU measurements and fused CUDA attention are pending. The CUDA command above runs the same explicit dequantize-then-attention algorithm and has not been executed in this environment.

## References

- G. Xiao et al. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. arXiv:2309.17453.
- T. Dettmers et al. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022 (related outlier background). arXiv:2208.07339.
- P. Micikevicius et al. *FP8 Formats for Deep Learning.* 2022. arXiv:2209.05433.
- T. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023 (long-context attention background). arXiv:2307.08691.

## License

MIT
