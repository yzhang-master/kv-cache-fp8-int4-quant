import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .cache import QuantizedKVCache


def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("attention expects [batch, heads, sequence, dim] tensors")
    q_float = q.to(torch.float32)
    k_float = k.to(torch.float32)
    v_float = v.to(torch.float32)
    scores = q_float @ k_float.transpose(-2, -1) / math.sqrt(q.shape[-1])
    query_length, key_length = scores.shape[-2:]
    query_positions = torch.arange(query_length, device=q.device).unsqueeze(1)
    key_positions = torch.arange(key_length, device=k.device).unsqueeze(0)
    allowed = key_positions <= query_positions + key_length - query_length
    scores = scores.masked_fill(~allowed, float("-inf"))
    probabilities = scores.softmax(dim=-1)
    return probabilities @ v_float, probabilities


def quantized_attention(q: Tensor, cache: QuantizedKVCache) -> tuple[Tensor, Tensor]:
    k, v = cache.dequantize()
    return causal_attention(q, k, v)


def attention_metrics(
    reference_output: Tensor,
    reference_probabilities: Tensor,
    candidate_output: Tensor,
    candidate_probabilities: Tensor,
    sink_tokens: int,
) -> dict[str, float]:
    difference = candidate_output - reference_output
    relative_error = difference.norm() / reference_output.norm().clamp_min(1e-12)
    cosine = F.cosine_similarity(reference_output, candidate_output, dim=-1).mean()
    reference_target = reference_probabilities.argmax(dim=-1)
    candidate_target = candidate_probabilities.argmax(dim=-1)
    target_agreement = (reference_target == candidate_target).to(torch.float32).mean()
    sink_mass_difference = (
        (
            candidate_probabilities[..., :sink_tokens].sum(dim=-1)
            - reference_probabilities[..., :sink_tokens].sum(dim=-1)
        )
        .abs()
        .mean()
    )
    return {
        "relative_output_error": relative_error.item(),
        "mean_output_cosine": cosine.item(),
        "attention_top1_agreement": target_agreement.item(),
        "mean_absolute_sink_mass_change": sink_mass_difference.item(),
    }


def make_long_context_proxy(
    sequence_length: int = 1024,
    heads: int = 4,
    head_dim: int = 32,
    sink_tokens: int = 4,
    seed: int = 71,
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    k = torch.randn(1, heads, sequence_length, head_dim, generator=generator)
    v = torch.randn(1, heads, sequence_length, head_dim, generator=generator)
    k[:, :, :sink_tokens] *= 8
    v[:, :, :sink_tokens] *= 3
    positions = torch.arange(sequence_length)
    targets = (positions * 37) % (positions + 1)
    q = k[:, :, targets] * 3 + 0.01 * torch.randn(
        1, heads, sequence_length, head_dim, generator=generator
    )
    return q, k, v
