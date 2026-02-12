#!/usr/bin/env python3
"""matmul → rmsnorm (projection → RMSNorm, LLaMA)"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
M = 4096


class MatMulOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.register_buffer('weight', torch.randn(H, H) * (2.0 / H) ** 0.5)
    def forward(self, x):
        return torch.matmul(x, self.weight)


class RMSNormOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.register_buffer('weight', torch.ones(H))
        self.eps = 1e-6
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class DependentPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = MatMulOp(H)
        self.op2 = RMSNormOp(H)
    def forward(self, x):
        return self.op2(self.op1(x))


def main():
    model = DependentPair().eval()
    x = torch.randn(M, H)
    workdir = f"/tmp/neuron_cross_matmul_rmsnorm_{M}x{H}"
    print(f"Compiling matmul→rmsnorm ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
    print("OK")


if __name__ == "__main__":
    main()
