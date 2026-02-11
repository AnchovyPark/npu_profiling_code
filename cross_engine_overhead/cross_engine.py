#!/usr/bin/env python3
"""
cross_engine.py - Cross-Engine Transition Overhead Profiling (AWS Inferentia2)

NeuronCore-v2의 엔진 간(TensorEngine ↔ VectorEngine) 전환 비용을 측정.
종속적(dependent) vs 독립적(independent) 연산 쌍을 비교하여
순수 전환 비용과 파이프라이닝 효과를 분리.

LLM 추론 과정에서 실제 발생하는 5가지 cross-engine 조합:
1. MatMul ↔ Add         (projection → residual connection)
2. MatMul ↔ GELU        (FFN up-projection → activation)
3. MatMul ↔ SiLU        (FFN gate-projection → activation, LLaMA style)
4. MatMul ↔ LayerNorm   (projection output → normalization)
5. MatMul ↔ Softmax     (QK^T → softmax in attention)

각 조합의 정방향/역방향 포함 → 총 10가지 조합
텐서 크기: 2048 ~ 4096 (5개 포인트)

Usage:
    python cross_engine.py
"""

import torch
import torch.nn as nn
import torch_neuronx
import time
import csv
import os
import numpy as np
from datetime import datetime


# ============================================================
# Configuration
# ============================================================
SIZES = [2048, 2560, 3072, 3584, 4096]
HIDDEN = 4096  # LLM hidden dimension (e.g., LLaMA 7B/8B)
ITERATIONS = 10
WARMUP = 3


# ============================================================
# FLOPs Calculation
# ============================================================
def calc_flops(op_name, M, H):
    """연산별 FLOPs 계산.
    
    Args:
        op_name: 연산 이름
        M: batch × seq_len (행 수)
        H: hidden dimension (열 수)
    """
    flops_map = {
        "matmul":    2 * M * H * H,    # (M, H) @ (H, H)
        "add":       M * H,            # element-wise add
        "gelu":      8 * M * H,        # x * 0.5 * (1 + tanh(...))
        "silu":      4 * M * H,        # x * sigmoid(x)
        "layernorm": 5 * M * H,        # mean, var, normalize, scale, shift
        "softmax":   5 * M * H,        # exp, sum, div
    }
    return flops_map[op_name]


# ============================================================
# Operation Modules
# ============================================================
class MatMulOp(nn.Module):
    """TensorEngine: 행렬곱 (H, H) weight"""
    def __init__(self, H):
        super().__init__()
        self.register_buffer('weight', torch.randn(H, H) * (2.0 / H) ** 0.5)

    def forward(self, x):
        return torch.matmul(x, self.weight)


class AddOp(nn.Module):
    """VectorEngine: element-wise add (bias)"""
    def __init__(self, H):
        super().__init__()
        self.register_buffer('bias', torch.randn(1, H) * 0.01)

    def forward(self, x):
        return x + self.bias


class GELUOp(nn.Module):
    """VectorEngine: GELU activation"""
    def forward(self, x):
        return torch.nn.functional.gelu(x)


class SiLUOp(nn.Module):
    """VectorEngine: SiLU activation (LLaMA)"""
    def forward(self, x):
        return torch.nn.functional.silu(x)


class LayerNormOp(nn.Module):
    """VectorEngine: Layer Normalization"""
    def __init__(self, H):
        super().__init__()
        self.ln = nn.LayerNorm(H)

    def forward(self, x):
        return self.ln(x)


class SoftmaxOp(nn.Module):
    """VectorEngine: Softmax"""
    def forward(self, x):
        return torch.nn.functional.softmax(x, dim=-1)


def make_op(name, H):
    """이름으로 Operation module 생성."""
    builders = {
        "matmul":    lambda: MatMulOp(H),
        "add":       lambda: AddOp(H),
        "gelu":      lambda: GELUOp(),
        "silu":      lambda: SiLUOp(),
        "layernorm": lambda: LayerNormOp(H),
        "softmax":   lambda: SoftmaxOp(),
    }
    return builders[name]()


# ============================================================
# Pair Models (Dependent / Independent)
# ============================================================
class DependentPair(nn.Module):
    """종속적 연산 쌍: op1의 출력이 op2의 입력.
    
    op1 → result → op2 (sequential dependency)
    Cross-engine 전환 비용이 latency에 직접 반영됨.
    """
    def __init__(self, op1, op2):
        super().__init__()
        self.op1 = op1
        self.op2 = op2

    def forward(self, x):
        y = self.op1(x)
        z = self.op2(y)
        return z.sum()  # 결과 사용 (NPU 연산 skip 방지)


class IndependentPair(nn.Module):
    """독립적 연산 쌍: 서로 다른 입력, 종속성 없음.
    
    op1(x1), op2(x2) (no dependency)
    엔진 간 병렬 실행 가능 → 전환 비용이 숨겨질 수 있음.
    """
    def __init__(self, op1, op2):
        super().__init__()
        self.op1 = op1
        self.op2 = op2

    def forward(self, x1, x2):
        y = self.op1(x1)
        z = self.op2(x2)
        return y.sum() + z.sum()  # 양쪽 결과 모두 사용


# ============================================================
# Benchmarking
# ============================================================
def benchmark(traced_model, inputs, iterations=ITERATIONS, warmup=WARMUP):
    """Traced model의 latency 측정 (ms).
    
    Args:
        traced_model: torch_neuronx.trace로 컴파일된 모델
        inputs: 입력 텐서 tuple
        iterations: 측정 반복 횟수
        warmup: 워밍업 횟수
    
    Returns:
        float: 평균 latency (ms)
    """
    # Warmup
    for _ in range(warmup):
        result = traced_model(*inputs)
        _ = result.item()  # sync

    # Measure
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = traced_model(*inputs)
        _ = result.item()  # NPU 연산 완료 보장 (sync)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # → ms

    return np.mean(latencies)


# ============================================================
# LLM Cross-Engine Operation Pairs
# ============================================================
# (op1, op2) — 실제 LLM 추론에서 발생하는 cross-engine 전환
PAIRS = [
    ("matmul", "add"),        # projection → residual add
    ("matmul", "gelu"),       # FFN up-projection → GELU (GPT style)
    ("matmul", "silu"),       # FFN gate-projection → SiLU (LLaMA style)
    ("matmul", "layernorm"),  # projection output → layer normalization
    ("matmul", "softmax"),    # QK^T → softmax (attention)
]

# Engine mapping (참고용)
ENGINE_MAP = {
    "matmul":    "TensorEngine",
    "add":       "VectorEngine",
    "gelu":      "VectorEngine",
    "silu":      "VectorEngine",
    "layernorm": "VectorEngine",
    "softmax":   "VectorEngine",
}


# ============================================================
# Main
# ============================================================
def main():
    results = []
    total_experiments = len(PAIRS) * 2 * len(SIZES)  # pairs × directions × sizes
    current = 0

    print("=" * 70)
    print("Cross-Engine Transition Overhead Profiling")
    print(f"  Sizes: {SIZES}")
    print(f"  Hidden dim: {HIDDEN}")
    print(f"  Iterations: {ITERATIONS} (warmup: {WARMUP})")
    print(f"  Pairs: {len(PAIRS)} × 2 directions × {len(SIZES)} sizes = {total_experiments}")
    print("=" * 70)

    for op1_base, op2_base in PAIRS:
        # 정방향: op1 → op2
        # 역방향: op2 → op1
        for (first, second) in [(op1_base, op2_base), (op2_base, op1_base)]:
            for size in SIZES:
                current += 1
                direction = f"{first}({ENGINE_MAP[first]}) → {second}({ENGINE_MAP[second]})"
                print(f"\n[{current}/{total_experiments}] {direction}, size={size}")

                # --- Dependent model ---
                dep_model = DependentPair(make_op(first, HIDDEN), make_op(second, HIDDEN))
                x_dep = torch.randn(size, HIDDEN)

                print(f"  Compiling dependent model...", end=" ", flush=True)
                try:
                    dep_traced = torch_neuronx.trace(dep_model, (x_dep,))
                    print("OK")
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

                print(f"  Benchmarking dependent...", end=" ", flush=True)
                dep_latency = benchmark(dep_traced, (x_dep,))
                print(f"{dep_latency:.3f} ms")

                # --- Independent model ---
                indep_model = IndependentPair(make_op(first, HIDDEN), make_op(second, HIDDEN))
                x1 = torch.randn(size, HIDDEN)
                x2 = torch.randn(size, HIDDEN)

                print(f"  Compiling independent model...", end=" ", flush=True)
                try:
                    indep_traced = torch_neuronx.trace(indep_model, (x1, x2))
                    print("OK")
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

                print(f"  Benchmarking independent...", end=" ", flush=True)
                indep_latency = benchmark(indep_traced, (x1, x2))
                print(f"{indep_latency:.3f} ms")

                # --- FLOPs ---
                flops_first = calc_flops(first, size, HIDDEN)
                flops_second = calc_flops(second, size, HIDDEN)

                # --- Summary ---
                diff = dep_latency - indep_latency
                print(f"  → Dependent: {dep_latency:.3f} ms | Independent: {indep_latency:.3f} ms | Diff: {diff:+.3f} ms")

                results.append({
                    "op1": first,
                    "op1_flops": flops_first,
                    "op2": second,
                    "op2_flops": flops_second,
                    "size": size,
                    "dep_latency_ms": round(dep_latency, 4),
                    "indep_latency_ms": round(indep_latency, 4),
                })

                # 메모리 정리
                del dep_model, dep_traced, indep_model, indep_traced
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ============================================================
    # Save CSV
    # ============================================================
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"cross_engine_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    fieldnames = [
        "op1",              # 시작 연산
        "op1_flops",        # 시작 연산의 FLOPs
        "op2",              # 끝 연산
        "op2_flops",        # 끝 연산의 FLOPs
        "size",             # 텐서 크기 (M dimension)
        "dep_latency_ms",   # 종속적 연산 latency (ms)
        "indep_latency_ms", # 독립적 연산 latency (ms)
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_path}")
    print(f"Total experiments: {len(results)}")
    print(f"{'=' * 70}")

    # 결과 요약 출력
    print("\n📊 Summary:")
    print(f"{'Op1':<12} {'Op2':<12} {'Size':>6} {'Dep(ms)':>10} {'Indep(ms)':>10} {'Diff(ms)':>10}")
    print("-" * 66)
    for r in results:
        diff = r["dep_latency_ms"] - r["indep_latency_ms"]
        print(f"{r['op1']:<12} {r['op2']:<12} {r['size']:>6} "
              f"{r['dep_latency_ms']:>10.3f} {r['indep_latency_ms']:>10.3f} {diff:>+10.3f}")


if __name__ == "__main__":
    main()
