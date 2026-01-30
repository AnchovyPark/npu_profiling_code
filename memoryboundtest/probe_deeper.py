#!/usr/bin/env python3
"""
PSUM Probe Deeper - Neuron Fusion 심층 분석
============================================

측정 항목:
1. 단일 matmul × 10 (예상 baseline)
2. 10번 독립적 matmul (병렬화 가능)
3. 10번 종속적 matmul (C1→C2→C3→...→C10)
"""

import time
import csv
import numpy as np
import torch
import torch_xla.core.xla_model as xm
from datetime import datetime


# 정사각형 텐서 크기들
SIZES = [256, 512, 1024, 2048, 4096]

ITERATIONS = 100
WARMUP = 10
NUM_MATMULS = 10  # 연속/독립 matmul 횟수


def sync():
    xm.mark_step()
    xm.wait_device_ops()


def measure_single(N):
    """단일 matmul latency 측정"""
    device = torch.device('xla:0')
    
    A = torch.randn(N, N, dtype=torch.float16, device=device)
    B = torch.randn(N, N, dtype=torch.float16, device=device)
    sync()
    
    # Warmup
    for _ in range(WARMUP):
        C = torch.matmul(A, B)
        _ = C.sum()
        sync()
    
    # Measure
    latencies = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        C = torch.matmul(A, B)
        _ = C.sum()
        sync()
        latencies.append((time.perf_counter() - start) * 1000)
    
    return np.mean(latencies), np.std(latencies)


def measure_independent(N):
    """10번 독립적 matmul (서로 다른 텐서, 병렬화 가능)"""
    device = torch.device('xla:0')
    
    # 10쌍의 독립적인 텐서
    tensors = [(torch.randn(N, N, dtype=torch.float16, device=device),
                torch.randn(N, N, dtype=torch.float16, device=device))
               for _ in range(NUM_MATMULS)]
    sync()
    
    # Warmup
    for _ in range(WARMUP):
        results = [torch.matmul(A, B) for A, B in tensors]
        _ = sum(r.sum() for r in results)
        sync()
    
    # Measure
    latencies = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        results = [torch.matmul(A, B) for A, B in tensors]
        _ = sum(r.sum() for r in results)
        sync()
        latencies.append((time.perf_counter() - start) * 1000)
    
    return np.mean(latencies), np.std(latencies)


def measure_chained(N):
    """10번 종속적 matmul (C1→C2→C3→...→C10)"""
    device = torch.device('xla:0')
    
    # 초기 텐서와 곱할 행렬들
    A = torch.randn(N, N, dtype=torch.float16, device=device)
    Bs = [torch.randn(N, N, dtype=torch.float16, device=device) 
          for _ in range(NUM_MATMULS)]
    sync()
    
    # Warmup
    for _ in range(WARMUP):
        C = A
        for B in Bs:
            C = torch.matmul(C, B)
        _ = C.sum()
        sync()
    
    # Measure
    latencies = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        C = A
        for B in Bs:
            C = torch.matmul(C, B)
        _ = C.sum()
        sync()
        latencies.append((time.perf_counter() - start) * 1000)
    
    return np.mean(latencies), np.std(latencies)


def main():
    print("🔬 PSUM Probe Deeper")
    print(f"   NUM_MATMULS = {NUM_MATMULS}")
    print(f"   ITERATIONS = {ITERATIONS}")
    print()
    
    results = []
    
    for N in SIZES:
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📐 현재 행렬 크기: {N} × {N}")
        print(f"   연산: ({N}, {N}) @ ({N}, {N}) → ({N}, {N})")
        print(f"   원소 개수: {N*N:,}개 | 크기: {N*N*2/1024/1024:.2f} MB")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # 1. 단일 matmul
        print(f"   [1/3] 단일 행렬곱 측정 중...", end=" ", flush=True)
        single_mean, single_std = measure_single(N)
        single_x10 = single_mean * NUM_MATMULS
        print(f"✓ {single_mean:.3f}ms (×{NUM_MATMULS} = {single_x10:.3f}ms)")
        
        # 2. 독립적 10회
        print(f"   [2/3] 독립 행렬곱 {NUM_MATMULS}회 측정 중...", end=" ", flush=True)
        indep_mean, indep_std = measure_independent(N)
        print(f"✓ {indep_mean:.3f}ms")
        
        # 3. 종속적 10회
        print(f"   [3/3] 연속 행렬곱 {NUM_MATMULS}회 측정 중...", end=" ", flush=True)
        chain_mean, chain_std = measure_chained(N)
        print(f"✓ {chain_mean:.3f}ms")
        
        # 비교
        print()
        print(f"   📊 결과 비교:")
        print(f"      단일×{NUM_MATMULS}:  {single_x10:.3f}ms (기준)")
        print(f"      독립 {NUM_MATMULS}회: {indep_mean:.3f}ms (기준 대비 {indep_mean/single_x10*100:.1f}%)")
        print(f"      연속 {NUM_MATMULS}회: {chain_mean:.3f}ms (기준 대비 {chain_mean/single_x10*100:.1f}%)")
        print(f"      연속-독립 차이: {chain_mean - indep_mean:+.3f}ms")
        print()
        
        results.append({
            'N': N,
            'tensor_bytes': N * N * 2,
            'single_ms': round(single_mean, 3),
            'single_std': round(single_std, 3),
            'single_x10_ms': round(single_x10, 3),
            'independent_ms': round(indep_mean, 3),
            'independent_std': round(indep_std, 3),
            'chained_ms': round(chain_mean, 3),
            'chained_std': round(chain_std, 3),
            'chain_vs_indep_ms': round(chain_mean - indep_mean, 3),
            'chain_vs_indep_pct': round((chain_mean - indep_mean) / indep_mean * 100, 2),
        })
    
    # CSV 저장
    filename = f"psum_deeper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ 저장: {filename}")


if __name__ == "__main__":
    main()
