#!/usr/bin/env python3
"""rmsnorm single independent (4096x4096)"""
import torch, torch.nn as nn, torch_neuronx
H, M = 4096, 4096

class RMSNormOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.ones(H, dtype=torch.bfloat16))
        self.eps = 1e-6
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

if __name__ == "__main__":
    model = RMSNormOp().eval()
    x = torch.randn(M, H, dtype=torch.bfloat16)
    print(f"Compiling rmsnorm single ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=f"/tmp/neuron_single_rmsnorm_{M}x{H}")
    print("OK")
