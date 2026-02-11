#!/usr/bin/env python3
"""
single_kernel/matmul.py - MatMul (TensorEngine) 단일 커널 프로파일링

LLM 추론에서 QKV projection, output projection, FFN up/down/gate projection에 해당.
(M, H) @ (H, H) 형태의 행렬곱.
"""

import torch
import torch.nn as nn
import torch_neuronx

H = 4096
SHAPES = [4096]  # M dimension (batch * seq_len)


class MatMulOp(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.register_buffer('weight', torch.randn(H, H) * (2.0 / H) ** 0.5)

    def forward(self, x):
        return torch.matmul(x, self.weight)


def main():
    print("=" * 50)
    print("MatMul Kernel Profiling - NEFF Generation")
    print(f"  H={H}, Shapes(M)={SHAPES}")
    print("=" * 50)

    for M in SHAPES:
        model = MatMulOp(H).eval()
        x = torch.randn(M, H)
        workdir = f"/tmp/neuron_single_matmul_{M}x{H}"

        print(f"  Compiling matmul ({M}, {H}) @ ({H}, {H})...", end=" ", flush=True)
        torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
        print("OK")

    print("Done.")


if __name__ == "__main__":
    main()
