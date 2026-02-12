#!/usr/bin/env python3
"""silu single independent (4096x4096)"""
import torch, torch.nn as nn, torch_neuronx
H, M = 4096, 4096

class SiLUOp(nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)

if __name__ == "__main__":
    model = SiLUOp().eval()
    x = torch.randn(M, H, dtype=torch.bfloat16)
    print(f"Compiling silu single ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=f"/tmp/neuron_single_silu_{M}x{H}")
    print("OK")
