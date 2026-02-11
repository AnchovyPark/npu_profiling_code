#!/usr/bin/env python3
"""
single_kernel/softmax.py - Softmax (VectorEngine) 단일 커널 프로파일링

LLM 추론에서 attention score → attention weight 변환에 해당.
"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
SHAPES = [4096]


class SoftmaxOp(nn.Module):
    def forward(self, x):
        return torch.nn.functional.softmax(x, dim=-1)


def main():
    print("=" * 50)
    print("Softmax Kernel Profiling - NEFF Generation")
    print(f"  H={H}, Shapes(M)={SHAPES}")
    print("=" * 50)

    for M in SHAPES:
        model = SoftmaxOp().eval()
        x = torch.randn(M, H)
        workdir = f"/tmp/neuron_single_softmax_{M}x{H}"

        print(f"  Compiling softmax ({M}, {H})...", end=" ", flush=True)
        torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
        print("OK")

    print("Done.")


if __name__ == "__main__":
    main()
