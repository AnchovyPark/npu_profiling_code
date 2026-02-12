#!/usr/bin/env python3
"""softmax single independent (4096x4096)"""
import torch, torch.nn as nn, torch_neuronx
H, M = 4096, 4096

class SoftmaxOp(nn.Module):
    def forward(self, x):
        return torch.nn.functional.softmax(x, dim=-1)

if __name__ == "__main__":
    model = SoftmaxOp().eval()
    x = torch.randn(M, H, dtype=torch.bfloat16)
    print(f"Compiling softmax single ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=f"/tmp/neuron_single_softmax_{M}x{H}")
    print("OK")
