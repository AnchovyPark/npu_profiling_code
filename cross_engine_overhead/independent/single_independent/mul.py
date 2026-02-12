#!/usr/bin/env python3
"""mul single independent (4096x4096)"""
import torch, torch.nn as nn, torch_neuronx
H, M = 4096, 4096

class MulOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('gate', torch.randn(1, H, dtype=torch.bfloat16) * 0.01)
    def forward(self, x):
        return x * self.gate

if __name__ == "__main__":
    model = MulOp().eval()
    x = torch.randn(M, H, dtype=torch.bfloat16)
    print(f"Compiling mul single ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=f"/tmp/neuron_single_mul_{M}x{H}")
    print("OK")
