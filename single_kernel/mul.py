#!/usr/bin/env python3
"""
single_kernel/mul.py - Element-wise Multiply (VectorEngine) 단일 커널 프로파일링

LLM 추론에서 SwiGLU의 gate * up_projection에 해당.
"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
SHAPES = [4096]


class MulOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.register_buffer('scale', torch.randn(1, H))

    def forward(self, x):
        return x * self.scale


def main():
    print("=" * 50)
    print("Multiply Kernel Profiling - NEFF Generation")
    print(f"  H={H}, Shapes(M)={SHAPES}")
    print("=" * 50)

    for M in SHAPES:
        model = MulOp(H).eval()
        x = torch.randn(M, H)
        workdir = f"/tmp/neuron_single_mul_{M}x{H}"

        print(f"  Compiling mul ({M}, {H})...", end=" ", flush=True)
        torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
        print("OK")

    print("Done.")


if __name__ == "__main__":
    main()
