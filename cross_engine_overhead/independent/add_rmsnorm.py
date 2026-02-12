#!/usr/bin/env python3
"""add + rmsnorm (independent)"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
M = 4096


class AddOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.register_buffer('bias', torch.randn(1, H) * 0.01)
    def forward(self, x):
        return x + self.bias

class RMSNormOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.register_buffer('weight', torch.ones(H))
        self.eps = 1e-6
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class IndependentPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = AddOp(H)
        self.op2 = RMSNormOp(H)

    def forward(self, x1, x2):
        return self.op1(x1), self.op2(x2)


def main():
    model = IndependentPair().eval()
    x1 = torch.randn(M, H)
    x2 = torch.randn(M, H)
    workdir = f"/tmp/neuron_indep_add_rmsnorm_{M}x{H}"
    print(f"Compiling add_rmsnorm independent ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x1, x2), compiler_workdir=workdir)
    print("OK")


if __name__ == "__main__":
    main()
