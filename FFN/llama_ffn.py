#!/usr/bin/env python3
"""
LLaMA 3.2 1B FFN (SwiGLU MLP)

Config (from HuggingFace):
  - hidden_size: 2048
  - intermediate_size: 8192
  - hidden_act: silu
  - mlp_bias: False

Flow:
  x (seq_len, hidden_size)
  → gate_proj(x)    matmul (2048 → 8192)  TensorEngine
  → SiLU(gate)      activation             VectorEngine
  → up_proj(x)      matmul (2048 → 8192)  TensorEngine
  → silu_gate * up   elementwise mul       VectorEngine
  → down_proj(...)  matmul (8192 → 2048)  TensorEngine

Note: gate_proj와 up_proj는 같은 입력 x를 받음 (independent)
      SiLU(gate) * up 이후 down_proj는 dependent
"""

import torch
import torch.nn as nn
import torch_neuronx

# ============================================================
# LLaMA 3.2 1B 설정
# ============================================================
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 8192
SEQ_LEN = 2048  # prefill 기준


class LLaMAFFN(nn.Module):
    """LLaMA SwiGLU FFN"""

    def __init__(self):
        super().__init__()
        # gate_proj: hidden → intermediate (no bias)
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        # up_proj: hidden → intermediate (no bias)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        # down_proj: intermediate → hidden (no bias)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, x):
        # x: (seq_len, hidden_size)

        # gate path: matmul → SiLU
        gate = torch.nn.functional.silu(self.gate_proj(x))

        # up path: matmul (independent from gate's SiLU)
        up = self.up_proj(x)

        # element-wise multiply
        hidden = gate * up

        # down projection
        output = self.down_proj(hidden)

        return output


def main():
    model = LLaMAFFN().to(torch.bfloat16).eval()

    x = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)

    workdir = f"/tmp/neuron_ffn_llama32_1b_s{SEQ_LEN}"

    print(f"Compiling LLaMA 3.2 1B FFN (SwiGLU)...")
    print(f"  hidden_size={HIDDEN_SIZE}, intermediate_size={INTERMEDIATE_SIZE}")
    print(f"  seq_len={SEQ_LEN}")
    print(f"  gate_proj: ({HIDDEN_SIZE} → {INTERMEDIATE_SIZE})")
    print(f"  up_proj:   ({HIDDEN_SIZE} → {INTERMEDIATE_SIZE})")
    print(f"  down_proj: ({INTERMEDIATE_SIZE} → {HIDDEN_SIZE})")

    torch_neuronx.trace(model, (x,), compiler_workdir=workdir)
    print("OK")


if __name__ == "__main__":
    main()
