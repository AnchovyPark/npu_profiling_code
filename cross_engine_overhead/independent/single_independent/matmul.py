#!/usr/bin/env python3
"""matmul single independent (4096x4096)"""
import torch, torch.nn as nn, torch_neuronx
H, M = 4096, 4096

class MatMulOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.randn(H, H, dtype=torch.bfloat16) * (2.0 / H) ** 0.5)
    def forward(self, x):
        return torch.matmul(x, self.weight)

if __name__ == "__main__":
    model = MatMulOp().eval()
    x = torch.randn(M, H, dtype=torch.bfloat16)
    print(f"Compiling matmul single ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=f"/tmp/neuron_single_matmul_{M}x{H}")
    print("OK")
