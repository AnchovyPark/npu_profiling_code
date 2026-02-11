#!/usr/bin/env python3
"""
single_kernel/silu.py - SiLU Activation (VectorEngine) 단일 커널 프로파일링

LLM 추론에서 SwiGLU의 gate activation (LLaMA 스타일)에 해당.
SiLU(x) = x * sigmoid(x)
"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
SHAPES = [4096]


class SiLUOp(nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)


def main():
    print("=" * 50)
    print("SiLU Kernel Profiling - NEFF Generation")
    print(f"  H={H}, Shapes(M)={SHAPES}")
    print("=" * 50)

    for M in SHAPES:
        model = SiLUOp().eval()
        x = torch.randn(M, H)
        workdir = f"/tmp/neuron_single_silu_{M}x{H}"

        print(f"  Compiling silu ({M}, {H})...", end=" ", flush=True)
        torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
        print("OK")

    print("Done.")


if __name__ == "__main__":
    main()
