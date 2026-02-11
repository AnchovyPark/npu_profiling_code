#!/usr/bin/env python3
"""
single_kernel/rope.py - Rotary Positional Embedding (VectorEngine) 단일 커널 프로파일링

LLM 추론에서 Q, K 텐서에 위치 정보를 주입하는 연산.
RoPE(x) = [x1*cos - x2*sin, x1*sin + x2*cos]
"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
SHAPES = [4096]


class RoPEOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        half_H = H // 2
        self.register_buffer('cos_val', torch.randn(1, half_H))
        self.register_buffer('sin_val', torch.randn(1, half_H))

    def forward(self, x):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([
            x1 * self.cos_val - x2 * self.sin_val,
            x1 * self.sin_val + x2 * self.cos_val
        ], dim=-1)


def main():
    print("=" * 50)
    print("RoPE Kernel Profiling - NEFF Generation")
    print(f"  H={H}, Shapes(M)={SHAPES}")
    print("=" * 50)

    for M in SHAPES:
        model = RoPEOp(H).eval()
        x = torch.randn(M, H)
        workdir = f"/tmp/neuron_single_rope_{M}x{H}"

        print(f"  Compiling rope ({M}, {H})...", end=" ", flush=True)
        torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
        print("OK")

    print("Done.")


if __name__ == "__main__":
    main()
