#!/usr/bin/env python3
"""matmul → silu (FFN gate → SiLU activation, LLaMA)"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
M = 4096


class MatMulOp(nn.Module):
    def __init__(self, H):
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
        self.op1 = MatMulOp(H)
        self.op2 = SiLUOp()
    def forward(self, x):
        return self.op2(self.op1(x))


def main():
    model = DependentPair().eval()
    x = torch.randn(M, H, dtype=torch.bfloat16)
    workdir = f"/tmp/neuron_cross_matmul_silu_{M}x{H}"
    print(f"Compiling matmul→silu ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
    print("OK")


if __name__ == "__main__":
    main()
