#!/usr/bin/env python3
"""gelu single independent (4096x4096)"""
import torch, torch.nn as nn, torch_neuronx
H, M = 4096, 4096

class GeLUOp(nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)

if __name__ == "__main__":
    model = GeLUOp().eval()
    x = torch.randn(M, H, dtype=torch.bfloat16)
    print(f"Compiling gelu single ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=f"/tmp/neuron_single_gelu_{M}x{H}")
    print("OK")
