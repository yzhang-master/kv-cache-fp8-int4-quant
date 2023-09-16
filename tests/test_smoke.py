import torch

from kvquant import __version__


def test_cpu_attention_smoke() -> None:
    assert __version__ == "0.1.0"
    x = torch.randn(1, 2, 8, 16, generator=torch.Generator().manual_seed(7))
    y = torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
