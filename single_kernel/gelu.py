#!/usr/bin/env python3
"""
single_kernel/gelu.py - GELU Activation (VectorEngine) 단일 커널 프로파일링

LLM 추론에서 FFN activation (GPT 스타일)에 해당.
"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
SHAPES = [4096]


class GELUOp(nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)


def main():
    print("=" * 50)
    print("GELU Kernel Profiling - NEFF Generation")
    print(f"  H={H}, Shapes(M)={SHAPES}")
    print("=" * 50)

    for M in SHAPES:
        model = GELUOp().eval()
        x = torch.randn(M, H)
        workdir = f"/tmp/neuron_single_gelu_{M}x{H}"

        print(f"  Compiling gelu ({M}, {H})...", end=" ", flush=True)
        torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
        print("OK")

    print("Done.")


if __name__ == "__main__":
    main()
