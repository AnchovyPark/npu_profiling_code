#!/usr/bin/env python3
"""layernorm + matmul (independent)"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
M = 4096


class LayerNormOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.norm = nn.LayerNorm(H)
    def forward(self, x):
        return self.norm(x)

class MatMulOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.register_buffer('weight', torch.randn(H, H) * (2.0 / H) ** 0.5)
    def forward(self, x):
        return torch.matmul(x, self.weight)

class IndependentPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = LayerNormOp(H)
        self.op2 = MatMulOp(H)

    def forward(self, x1, x2):
        return self.op1(x1), self.op2(x2)


def main():
    model = IndependentPair().eval()
    x1 = torch.randn(M, H)
    x2 = torch.randn(M, H)
    workdir = f"/tmp/neuron_indep_layernorm_matmul_{M}x{H}"
    print(f"Compiling layernorm_matmul independent ({M},{H})...", end=" ", flush=True)
    torch_neuronx.trace(model, (x1, x2), compiler_workdir=workdir)
    print("OK")


if __name__ == "__main__":
    main()
