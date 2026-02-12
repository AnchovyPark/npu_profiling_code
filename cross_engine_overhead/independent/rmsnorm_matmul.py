#!/usr/bin/env python3
"""rmsnorm + matmul (independent)"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
M = 4096


class RMSNormOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.register_buffer('weight', torch.ones(H))
        self.eps = 1e-6
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class MatMulOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.register_buffer('weight', torch.randn(H, H) * (2.0 / H) ** 0.5)
    def forward(self, x):
        return torch.matmul(x, self.weight)

class IndependentPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = RMSNormOp(H)
        self.op2 = MatMulOp(H)

    def forward(self, x1, x2):
        return self.op1(x1), self.op2(x2)


def main():
    model = IndependentPair().eval()
    x1 = torch.randn(M, H)
    x2 = torch.randn(M, H)
    workdir = f"/tmp/neuron_indep_rmsnorm_matmul_{M}x{H}"
    print(f"Compiling rmsnorm_matmul independent ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x1, x2), compiler_workdir=workdir)
    print("OK")


if __name__ == "__main__":
    main()
