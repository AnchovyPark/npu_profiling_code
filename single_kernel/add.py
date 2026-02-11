#!/usr/bin/env python3
"""
single_kernel/add.py - Add (VectorEngine) 단일 커널 프로파일링

LLM 추론에서 residual connection (x + projection_output)에 해당.
"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
SHAPES = [4096]


class AddOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.register_buffer('bias', torch.randn(1, H) * 0.01)

    def forward(self, x):
        return x + self.bias


def main():
    print("=" * 50)
    print("Add Kernel Profiling - NEFF Generation")
    print(f"  H={H}, Shapes(M)={SHAPES}")
    print("=" * 50)

    for M in SHAPES:
        model = AddOp(H).eval()
        x = torch.randn(M, H)
        workdir = f"/tmp/neuron_single_add_{M}x{H}"

        print(f"  Compiling add ({M}, {H})...", end=" ", flush=True)
        torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
        print("OK")

    print("Done.")


if __name__ == "__main__":
    main()
