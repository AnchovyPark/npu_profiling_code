#!/usr/bin/env python3
"""layernorm single independent (4096x4096)"""
import torch, torch.nn as nn, torch_neuronx
H, M = 4096, 4096

class LayerNormOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(H)
    def forward(self, x):
        return self.norm(x)

if __name__ == "__main__":
    model = LayerNormOp().to(torch.bfloat16).eval()
    x = torch.randn(M, H, dtype=torch.bfloat16)
    print(f"Compiling layernorm single ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=f"/tmp/neuron_single_layernorm_{M}x{H}")
    print("OK")
