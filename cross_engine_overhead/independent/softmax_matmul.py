#!/usr/bin/env python3
"""softmax + matmul (independent)"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
M = 4096


class SoftmaxOp(nn.Module):
    def forward(self, x):
        return torch.nn.functional.softmax(x, dim=-1)

class MatMulOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.register_buffer('weight', torch.randn(H, H, dtype=torch.bfloat16) * (2.0 / H) ** 0.5)
    def forward(self, x):
        return torch.matmul(x, self.weight)

class IndependentPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = SoftmaxOp()
        self.op2 = MatMulOp(H)

    def forward(self, x1, x2):
        return self.op1(x1), self.op2(x2)


def main():
    model = IndependentPair().eval()
    x1 = torch.randn(M, H, dtype=torch.bfloat16)
    x2 = torch.randn(M, H, dtype=torch.bfloat16)
    workdir = f"/tmp/neuron_indep_softmax_matmul_{M}x{H}"
    print(f"Compiling softmax_matmul independent ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x1, x2), compiler_workdir=workdir)
    print("OK")


if __name__ == "__main__":
    main()
