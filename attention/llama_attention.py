#!/usr/bin/env python3
"""
LLaMA-style Attention (QKV 완성 후 attention 연산만)

Flow:
  Q, K, V (이미 projection 완료된 텐서)
  → Q @ K^T                    (matmul, TensorEngine)
  → / sqrt(head_dim)           (scalar div, VectorEngine)
  → softmax                    (VectorEngine)
  → attn_weights @ V           (matmul, TensorEngine)

LLaMA-7B 기준 파라미터:
  - num_heads = 32
  - head_dim = 128
  - hidden_dim = 4096

입력 shape: (batch=1, num_heads, seq_len, head_dim)
"""

import torch
import torch.nn as nn
import torch_neuronx
import math

# ============================================================
# LLaMA-7B 기준 설정
# ============================================================
NUM_HEADS = 32
HEAD_DIM = 128
SEQ_LEN = 2048  # prefill 기준


class LLaMAAttention(nn.Module):
    """QKV가 이미 완성된 상태에서 attention 연산만 수행"""

    def __init__(self):
        super().__init__()
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

    def forward(self, q, k, v):
        # q, k, v: (1, num_heads, seq_len, head_dim)

        # Q @ K^T → (1, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # scale
        attn_scores = attn_scores * self.scale

        # softmax
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        # attn_weights @ V → (1, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)

        return output


def main():
    model = LLaMAAttention().eval()

    # 랜덤 QKV 텐서 생성
    q = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)

    workdir = f"/tmp/neuron_attention_h{NUM_HEADS}_s{SEQ_LEN}_d{HEAD_DIM}"

    print(f"Compiling LLaMA attention...")
    print(f"  num_heads={NUM_HEADS}, seq_len={SEQ_LEN}, head_dim={HEAD_DIM}")
    print(f"  Q/K/V shape: (1, {NUM_HEADS}, {SEQ_LEN}, {HEAD_DIM})")
    print(f"  attn_scores shape: (1, {NUM_HEADS}, {SEQ_LEN}, {SEQ_LEN})")

    torch_neuronx.trace(model, (q, k, v), compiler_workdir=workdir)
    print("OK")


if __name__ == "__main__":
    main()
