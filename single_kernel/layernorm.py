#!/usr/bin/env python3
"""
single_kernel/layernorm.py - LayerNorm (VectorEngine) 단일 커널 프로파일링

LLM 추론에서 normalization (GPT 스타일)에 해당.
"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
SHAPES = [4096]


class LayerNormOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.ln = nn.LayerNorm(H)

    def forward(self, x):
        return self.ln(x)


def main():
    print("=" * 50)
    print("LayerNorm Kernel Profiling - NEFF Generation")
    print(f"  H={H}, Shapes(M)={SHAPES}")
    print("=" * 50)

    for M in SHAPES:
        model = LayerNormOp(H).eval()
        x = torch.randn(M, H)
        workdir = f"/tmp/neuron_single_layernorm_{M}x{H}"

        print(f"  Compiling layernorm ({M}, {H})...", end=" ", flush=True)
        torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
        print("OK")

    print("Done.")


if __name__ == "__main__":
    main()
