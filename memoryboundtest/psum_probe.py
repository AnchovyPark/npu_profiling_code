#!/usr/bin/env python3
"""
PSUM Probe - Neuron PSUM/Fusion 효과 측정
"""

import time
import csv
import numpy as np
import torch
import torch_xla.core.xla_model as xm
from datetime import datetime


# 설정
# LLaMA 3 8B FFN 구조 기준
# (batch_seq, 4096) @ (4096, 14336) = C1 @ (14336, 4096) = C2
# A(M×K1) @ B(K1×K2) = C1(M×K2) @ D(K2×N) = C2(M×N)

HIDDEN = 4096
INTERMEDIATE = 14336

# Bucket 크기 정의
BUCKET_SIZES = [1024, 2048, 4096, 8192]

# Config 자동 생성 (M, K1, K2, N) = (batch_seq, hidden, intermediate, hidden)
CONFIGS = []
for i, bucket in enumerate(BUCKET_SIZES):
    prev_bucket = BUCKET_SIZES[i-1] if i > 0 else 0

    # 각 bucket에서 테스트할 크기들
    test_sizes = [
        prev_bucket + 1,              # 이전 bucket 바로 다음
        (prev_bucket + bucket) // 2,  # 중간값
        bucket,                       # bucket 크기
    ]

    for batch_seq in test_sizes:
        CONFIGS.append((batch_seq, HIDDEN, INTERMEDIATE, HIDDEN))
ITERATIONS = 100
WARMUP = 10
BYTES_PER_ELEM = 2  # float16


def sync():
    xm.mark_step()
    xm.wait_device_ops()


def calc_arithmetic_intensity(M, K1, K2, N, fused=True):
    """
    연속 matmul의 Arithmetic Intensity 계산
    
    A(M×K1) @ B(K1×K2) = C1(M×K2)
    C1(M×K2) @ D(K2×N) = C2(M×N)
    
    Args:
        fused: True면 C1이 PSUM에 머묾, False면 C1 write-back
    
    Returns:
        dict: flops, bytes, arithmetic_intensity
    """
    # FLOPs: 두 번의 matmul
    flops = 2 * M * K1 * K2 + 2 * M * K2 * N
    
    # Bytes moved
    if fused:
        # A + B + D + C2 (C1은 PSUM에 머묾)
        bytes_moved = (M * K1 + K1 * K2 + K2 * N + M * N) * BYTES_PER_ELEM
    else:
        # A + B + C1(write) + C1(read) + D + C2
        bytes_moved = (M * K1 + K1 * K2 + M * K2 + M * K2 + K2 * N + M * N) * BYTES_PER_ELEM
    
    ai = flops / bytes_moved
    
    return {
        'flops': flops,
        'bytes_moved': bytes_moved,
        'arithmetic_intensity': round(ai, 3)
    }


def measure_fusion(M, K1, K2, N):
    """개별 vs 연속 연산 비교"""
    device = torch.device('xla:0')

    A = torch.randn(M, K1, dtype=torch.float16, device=device)
    B = torch.randn(K1, K2, dtype=torch.float16, device=device)
    X = torch.randn(M, K2, dtype=torch.float16, device=device)  # Fixed: (M, K2) for fair comparison
    D = torch.randn(K2, N, dtype=torch.float16, device=device)
    sync()

    # 개별 (독립적인 두 matmul, 같은 shape)
    for _ in range(WARMUP):
        E1 = torch.matmul(A, B)   # (M, K1) @ (K1, K2) = (M, K2)
        E2 = torch.matmul(X, D)   # (M, K2) @ (K2, N) = (M, N)
        _ = E1.sum() + E2.sum()
        sync()

    separate = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        E1 = torch.matmul(A, B)   # (M, K1) @ (K1, K2) = (M, K2)
        E2 = torch.matmul(X, D)   # (M, K2) @ (K2, N) = (M, N)
        _ = E1.sum() + E2.sum()
        sync()
        separate.append((time.perf_counter() - start) * 1000)
    
    # 연속 (한번에)
    for _ in range(WARMUP):
        C1 = torch.matmul(A, B)
        C2 = torch.matmul(C1, D)
        _ = C2.sum()
        sync()
    
    fused = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        C1 = torch.matmul(A, B)
        C2 = torch.matmul(C1, D)
        _ = C2.sum()
        sync()
        fused.append((time.perf_counter() - start) * 1000)
    
    sep_mean, sep_std = np.mean(separate), np.std(separate)
    fused_mean, fused_std = np.mean(fused), np.std(fused)
    diff = sep_mean - fused_mean
    
    # AI 계산
    ai_fused = calc_arithmetic_intensity(M, K1, K2, N, fused=True)
    ai_separate = calc_arithmetic_intensity(M, K1, K2, N, fused=False)
    
    # intermediate tensor C1 크기
    intermediate_size = M * K2
    intermediate_bytes = intermediate_size * BYTES_PER_ELEM
    
    return {
        'M': M,
        'K1': K1,
        'K2': K2,
        'N': N,
        'intermediate_size': intermediate_size,
        'intermediate_bytes': intermediate_bytes,
        'flops': ai_fused['flops'],
        'bytes_fused': ai_fused['bytes_moved'],
        'bytes_separate': ai_separate['bytes_moved'],
        'ai_fused': ai_fused['arithmetic_intensity'],
        'ai_separate': ai_separate['arithmetic_intensity'],
        'separate_ms': round(sep_mean, 3),
        'separate_std': round(sep_std, 3),
        'fused_ms': round(fused_mean, 3),
        'fused_std': round(fused_std, 3),
        'difference_ms': round(diff, 3),
        'fusion_effective': fused_mean < sep_mean * 0.9
    }


def main():
    print("🔬 PSUM Probe\n")
    
    results = []
    for M, K1, K2, N in CONFIGS:
        print(f"  ({M}, {K1}) @ ({K1}, {K2}) @ ({K2}, {N})...", end=" ", flush=True)
        r = measure_fusion(M, K1, K2, N)
        status = "✓" if r['fusion_effective'] else "✗"
        int_mb = r['intermediate_bytes'] / (1024*1024)
        print(f"{status} C1={int_mb:.1f}MB, sep={r['separate_ms']:.2f}ms, fused={r['fused_ms']:.2f}ms, diff={r['difference_ms']:+.2f}ms, AI={r['ai_fused']:.1f}")
        results.append(r)
    
    # CSV 저장
    filename = f"psum_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ 저장: {filename}")


if __name__ == "__main__":
    main()
