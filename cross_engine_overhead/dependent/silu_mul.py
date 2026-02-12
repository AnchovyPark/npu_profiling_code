#!/usr/bin/env python3
"""silu → mul (SiLU activation → gate mul, SwiGLU)"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
M = 4096


class SiLUOp(nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)


class MulOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.register_buffer('gate', torch.randn(1, H, dtype=torch.bfloat16) * 0.01)
    def forward(self, x):
        return x * self.gate


class DependentPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = SiLUOp()
        self.op2 = MulOp(H)
    def forward(self, x):
        return self.op2(self.op1(x))


def main():
    model = DependentPair().eval()
    x = torch.randn(M, H, dtype=torch.bfloat16)
    workdir = f"/tmp/neuron_cross_silu_mul_{M}x{H}"
    print(f"Compiling silu→mul ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
    print("OK")


if __name__ == "__main__":
    main()
