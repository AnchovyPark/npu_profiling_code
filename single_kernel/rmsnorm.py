#!/usr/bin/env python3
"""
single_kernel/rmsnorm.py - RMSNorm (VectorEngine) 단일 커널 프로파일링

LLM 추론에서 pre-attention / pre-FFN normalization (LLaMA 스타일)에 해당.
RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
SHAPES = [4096]


class RMSNormOp(nn.Module):
    def __init__(self, H, eps=1e-6):
        super().__init__()
        self.register_buffer('weight', torch.ones(H))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def main():
    print("=" * 50)
    print("RMSNorm Kernel Profiling - NEFF Generation")
    print(f"  H={H}, Shapes(M)={SHAPES}")
    print("=" * 50)

    for M in SHAPES:
        model = RMSNormOp(H).eval()
        x = torch.randn(M, H)
        workdir = f"/tmp/neuron_single_rmsnorm_{M}x{H}"

        print(f"  Compiling rmsnorm ({M}, {H})...", end=" ", flush=True)
        torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
        print("OK")

    print("Done.")


if __name__ == "__main__":
    main()
