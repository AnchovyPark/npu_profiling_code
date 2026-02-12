#!/usr/bin/env python3
"""matmul → silu (dependent, 4096x4096)"""
import torch, torch.nn as nn, torch_neuronx
H, M = 4096, 4096

class MatMulOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.randn(H, H, dtype=torch.bfloat16) * (2.0 / H) ** 0.5)
    def forward(self, x):
        return torch.matmul(x, self.weight)

class SiLUOp(nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)

class DependentPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = MatMulOp()
        self.op2 = SiLUOp()
    def forward(self, x):
        return self.op2(self.op1(x))

if __name__ == "__main__":
    model = DependentPair().eval()
    x = torch.randn(M, H, dtype=torch.bfloat16)
    print(f"Compiling matmul→silu dependent ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=f"/tmp/neuron_dep_matmul_silu_{M}x{H}")
    print("OK")
