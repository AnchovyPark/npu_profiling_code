#!/usr/bin/env python3
"""matmul + add (independent, 8192x8192)"""
import torch, torch.nn as nn, torch_neuronx
H, M = 8192, 8192

class MatMulOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.randn(H, H) * (2.0 / H) ** 0.5)
    def forward(self, x):
        return torch.matmul(x, self.weight)

class AddOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('bias', torch.randn(1, H) * 0.01)
    def forward(self, x):
        return x + self.bias

class IndependentPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = MatMulOp()
        self.op2 = AddOp()
    def forward(self, x1, x2):
        return self.op1(x1), self.op2(x2)

if __name__ == "__main__":
    model = IndependentPair().eval()
    x1, x2 = torch.randn(M, H), torch.randn(M, H)
    print(f"Compiling matmul+add independent ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x1, x2), compiler_workdir=f"/tmp/neuron_indep_matmul_add_{M}x{H}")
    print("OK")
