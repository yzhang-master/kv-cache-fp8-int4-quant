from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


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
        raise ValueError("INT4 KV quantization requires finite inputs")


def _parameters(x: Tensor, granularity: str) -> tuple[Tensor, Tensor]:
    dimensions = (-2, -1) if granularity == "head" else (-1,)
    minimum = x.amin(dim=dimensions, keepdim=True)
    maximum = x.amax(dim=dimensions, keepdim=True)
    scale = (maximum - minimum) / 15.0
    constant = scale < torch.finfo(scale.dtype).eps
    constant_scale = torch.where(minimum == 0, torch.ones_like(minimum), minimum.abs() / 15.0)
    scale = torch.where(constant, constant_scale, scale)
    zero = torch.round(-minimum / scale).clamp(0, 15)
    negative_constant = constant & (minimum < 0)
    zero = torch.where(negative_constant, torch.full_like(zero, 15), zero)
    zero = torch.where(constant & ~negative_constant, torch.zeros_like(zero), zero)
    return scale, zero


def _pack(q: Tensor) -> Tensor:
    if q.shape[-1] % 2:
        q = F.pad(q, (0, 1))
    q = q.to(torch.uint8)
    return q[..., 0::2] | (q[..., 1::2] << 4)


def _unpack(packed: Tensor, head_dim: int) -> Tensor:
    low = packed & 0x0F
    high = packed >> 4
    return torch.stack((low, high), dim=-1).flatten(-2)[..., :head_dim]


@dataclass
class Int4KVCache:
    k_data: Tensor
    v_data: Tensor
    k_scale: Tensor
    v_scale: Tensor
    k_zero: Tensor
    v_zero: Tensor
    k_sink: Tensor
    v_sink: Tensor
    granularity: str
    head_dim: int
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
                self.k_zero,
                self.v_zero,
                self.k_sink,
                self.v_sink,
            )
        )

    def dequantize(self) -> tuple[Tensor, Tensor]:
        k_q = _unpack(self.k_data, self.head_dim).to(torch.float32)
        v_q = _unpack(self.v_data, self.head_dim).to(torch.float32)
        k = ((k_q - self.k_zero.to(torch.float32)) * self.k_scale).to(self.dtype)
        v = ((v_q - self.v_zero.to(torch.float32)) * self.v_scale).to(self.dtype)
        return torch.cat((self.k_sink, k), dim=2), torch.cat((self.v_sink, v), dim=2)


def quantize_int4(
    k: Tensor, v: Tensor, granularity: str = "token", sink_tokens: int = 4
) -> Int4KVCache:
    _validate_inputs(k, v, granularity, sink_tokens)
    k_non_sink = k[:, :, sink_tokens:].detach().to(torch.float32).contiguous()
    v_non_sink = v[:, :, sink_tokens:].detach().to(torch.float32).contiguous()
    if granularity == "head" and k_non_sink.shape[2] == 0:
        scale_shape = (k.shape[0], k.shape[1], 1, 1)
        k_scale = torch.ones(scale_shape)
        v_scale = torch.ones(scale_shape)
        k_zero = torch.zeros(scale_shape)
        v_zero = torch.zeros(scale_shape)
    else:
        k_scale, k_zero = _parameters(k_non_sink, granularity)
        v_scale, v_zero = _parameters(v_non_sink, granularity)
    k_q = (k_non_sink / k_scale + k_zero).round().clamp(0, 15)
    v_q = (v_non_sink / v_scale + v_zero).round().clamp(0, 15)
    return Int4KVCache(
        k_data=_pack(k_q),
        v_data=_pack(v_q),
        k_scale=k_scale,
        v_scale=v_scale,
        k_zero=k_zero.to(torch.uint8),
        v_zero=v_zero.to(torch.uint8),
        k_sink=k[:, :, :sink_tokens].detach().clone(),
        v_sink=v[:, :, :sink_tokens].detach().clone(),
        granularity=granularity,
        head_dim=k.shape[-1],
        dtype=k.dtype,
    )
