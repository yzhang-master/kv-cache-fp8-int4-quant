from dataclasses import dataclass

import torch
from torch import Tensor

FP8_E4M3_MAX = 448.0


def _validate_inputs(k: Tensor, v: Tensor, granularity: str, sink_tokens: int) -> None:
    if k.shape != v.shape or k.ndim != 4 or not k.is_floating_point():
        raise ValueError(
            "K and V must be matching floating-point [batch, heads, sequence, dim] tensors"
        )
    if granularity not in {"head", "token"}:
        raise ValueError("granularity must be head or token")
    if not 0 <= sink_tokens <= k.shape[2]:
        raise ValueError("sink_tokens must be between zero and sequence length")
    if not torch.isfinite(k).all() or not torch.isfinite(v).all():
        raise ValueError("FP8 KV quantization requires finite inputs")


def _scale(x: Tensor, granularity: str) -> Tensor:
    dimensions = (-2, -1) if granularity == "head" else (-1,)
    absmax = x.abs().amax(dim=dimensions, keepdim=True)
    return torch.where(absmax > 0, absmax / FP8_E4M3_MAX, torch.ones_like(absmax))


@dataclass
class FP8KVCache:
    k_data: Tensor
    v_data: Tensor
    k_scale: Tensor
    v_scale: Tensor
    k_sink: Tensor
    v_sink: Tensor
    granularity: str
    dtype: torch.dtype

    @property
    def sink_tokens(self) -> int:
        return self.k_sink.shape[2]

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor.nbytes
            for tensor in (
                self.k_data,
                self.v_data,
                self.k_scale,
                self.v_scale,
                self.k_sink,
                self.v_sink,
            )
        )

    def dequantize(self) -> tuple[Tensor, Tensor]:
        k = (self.k_data.to(torch.float32) * self.k_scale).to(self.dtype)
        v = (self.v_data.to(torch.float32) * self.v_scale).to(self.dtype)
        return torch.cat((self.k_sink, k), dim=2), torch.cat((self.v_sink, v), dim=2)


def quantize_fp8(
    k: Tensor, v: Tensor, granularity: str = "token", sink_tokens: int = 4
) -> FP8KVCache:
    _validate_inputs(k, v, granularity, sink_tokens)
    k_non_sink = k[:, :, sink_tokens:].detach().to(torch.float32).contiguous()
    v_non_sink = v[:, :, sink_tokens:].detach().to(torch.float32).contiguous()
    if granularity == "head" and k_non_sink.shape[2] == 0:
        scale_shape = (k.shape[0], k.shape[1], 1, 1)
        k_scale = torch.ones(scale_shape)
        v_scale = torch.ones(scale_shape)
    else:
        k_scale = _scale(k_non_sink, granularity)
        v_scale = _scale(v_non_sink, granularity)
    k_data = (k_non_sink / k_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    v_data = (v_non_sink / v_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return FP8KVCache(
        k_data=k_data,
        v_data=v_data,
        k_scale=k_scale,
        v_scale=v_scale,
        k_sink=k[:, :, :sink_tokens].detach().clone(),
        v_sink=v[:, :, :sink_tokens].detach().clone(),
        granularity=granularity,
        dtype=k.dtype,
    )
