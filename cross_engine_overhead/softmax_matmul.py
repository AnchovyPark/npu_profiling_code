#!/usr/bin/env python3
"""softmax → matmul (Softmax → Attention×V matmul)"""

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
        self.register_buffer('weight', torch.randn(H, H) * (2.0 / H) ** 0.5)
    def forward(self, x):
        return torch.matmul(x, self.weight)


class DependentPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = SoftmaxOp()
        self.op2 = MatMulOp(H)
    def forward(self, x):
        return self.op2(self.op1(x))


def main():
    model = DependentPair().eval()
    x = torch.randn(M, H)
    workdir = f"/tmp/neuron_cross_softmax_matmul_{M}x{H}"
    print(f"Compiling softmax→matmul ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
    print("OK")


if __name__ == "__main__":
    main()
