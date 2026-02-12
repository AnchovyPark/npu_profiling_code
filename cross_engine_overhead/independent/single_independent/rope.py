#!/usr/bin/env python3
"""rope single independent (4096x4096)"""
import torch, torch.nn as nn, torch_neuronx
H, M = 4096, 4096

class RoPEOp(nn.Module):
    def __init__(self):
        super().__init__()
        half = H // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(M, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        self.register_buffer('cos_cached', torch.cos(angles).to(torch.bfloat16))
        self.register_buffer('sin_cached', torch.sin(angles).to(torch.bfloat16))

    def forward(self, x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        out1 = x1 * self.cos_cached - x2 * self.sin_cached
        out2 = x1 * self.sin_cached + x2 * self.cos_cached
        return torch.cat([out1, out2], dim=-1)

if __name__ == "__main__":
    model = RoPEOp().eval()
    x = torch.randn(M, H, dtype=torch.bfloat16)
    print(f"Compiling rope single ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=f"/tmp/neuron_single_rope_{M}x{H}")
    print("OK")
