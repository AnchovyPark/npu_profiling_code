"""
Bucketing Efficiency Profiler for vLLM-Neuron

목적: 같은 bucket 내의 다양한 입력 크기가 실제로 동일한 latency를 보이는지 검증
예: 65, 80, 100, 120, 127, 128 토큰 입력이 모두 128 bucket을 사용할 때 latency 비교

두 가지 측정 모드:
1. NPU-only: 순수 NPU matmul 연산 시간만 측정 (padding 미리 완료)
2. E2E Pipeline: bucket 선택 + padding + 전송 + 연산 전체 측정
"""

import torch
import torch_neuronx
import torch.nn.functional as F
import time
import csv
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


class BucketManager:
    """Bucket 관리 및 선택 로직 (vLLM-Neuron 방식)"""

    DEFAULT_BUCKETS = [128, 256, 512, 1024, 2048, 4096]

    def __init__(self, buckets: Optional[List[int]] = None):
        self.buckets = sorted(buckets if buckets else self.DEFAULT_BUCKETS)

    def get_bucket(self, size: int) -> int:
        """주어진 크기에 대해 가장 가까운 bucket 반환"""
        for bucket in self.buckets:
            if size <= bucket:
                return bucket
        return self.buckets[-1]

    def get_padding_size(self, size: int) -> int:
        """필요한 padding 크기 반환"""
        return self.get_bucket(size) - size


class MatmulModule(torch.nn.Module):
    """단순 Matmul 모듈"""
    def forward(self, a, b):
        return torch.matmul(a, b)


@dataclass
class MeasurementResult:
    """측정 결과 저장"""
    actual_size: int
    bucket_size: int
    padding_size: int
    latency_ms: float
    latency_std_ms: float
    mode: str  # 'npu_only' or 'e2e'


def compile_bucket_models(
    M: int, 
    K: int, 
    bucket_sizes: List[int], 
    dtype: torch.dtype = torch.float16
) -> Dict[int, torch.jit.ScriptModule]:
    """
    각 bucket 크기에 대해 모델 컴파일
    
    vLLM-Neuron에서는 K(hidden_dim)는 고정, M(seq_len)만 bucketing
    여기서는 간단히 N 차원을 bucketing하는 것으로 시뮬레이션
    """
    compiled_models = {}
    model = MatmulModule().eval()

    for bucket_N in bucket_sizes:
        print(f"  [컴파일] Shape: ({M}, {K}) x ({K}, {bucket_N})", end='', flush=True)
        
        A = torch.randn(M, K, dtype=dtype)
        B = torch.randn(K, bucket_N, dtype=dtype)
        
        workdir = f"/tmp/neuron_bucket_{M}x{K}x{bucket_N}"
        compiled_model = torch_neuronx.trace(model, (A, B), compiler_workdir=workdir)
        compiled_models[bucket_N] = compiled_model
        
        print(" ✓")

    return compiled_models


def measure_npu_only(
    compiled_model,
    M: int, K: int, bucket_N: int,
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iterations: int = 100
) -> Tuple[float, float]:
    """
    순수 NPU 연산 시간만 측정 (padding 오버헤드 제외)
    
    텐서를 bucket 크기로 미리 생성하여 padding 시간 제외
    """
    device = torch.device('xla:0')
    import torch_xla.core.xla_model as xm

    # Bucket 크기로 텐서 생성 (padding 이미 완료된 상태)
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, bucket_N, dtype=dtype, device=device)
    
    # 동기화
    xm.mark_step()
    xm.wait_device_ops()

    # Warmup
    for _ in range(warmup):
        results = compiled_model(A, B)
        _ = results.sum()  # 강제로 결과 사용
        xm.mark_step()
        xm.wait_device_ops()

    # 측정
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        results = compiled_model(A, B)
        _ = results.sum()  # 강제로 결과 사용
        xm.mark_step()
        xm.wait_device_ops()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return np.mean(latencies), np.std(latencies)


def measure_padding_only(
    M: int, 
    actual_K: int, 
    bucket_K: int, 
    N: int,
    dtype: torch.dtype = torch.float16,
    iterations: int = 1000
) -> Tuple[float, float]:
    """
    CPU에서 Padding 시간만 측정
    
    vLLM 방식: CPU에서 F.pad로 padding 수행
    """
    latencies = []
    
    for _ in range(iterations):
        # 실제 크기로 텐서 생성
        A = torch.randn(M, actual_K, dtype=dtype)
        B = torch.randn(actual_K, N, dtype=dtype)
        
        start = time.perf_counter()
        # CPU에서 Padding (vLLM 방식)
        if actual_K < bucket_K:
            A_padded = F.pad(A, (0, bucket_K - actual_K), value=0.0)
            B_padded = F.pad(B, (0, 0, 0, bucket_K - actual_K), value=0.0)
        else:
            A_padded = A
            B_padded = B
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)
    
    return np.mean(latencies), np.std(latencies)


def measure_transfer_only(
    M: int, 
    bucket_K: int, 
    N: int,
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iterations: int = 100
) -> Tuple[float, float]:
    """
    CPU → NPU 전송 시간만 측정 (padding 완료된 텐서)
    """
    device = torch.device('xla:0')
    import torch_xla.core.xla_model as xm
    
    # Warmup
    for _ in range(warmup):
        A = torch.randn(M, bucket_K, dtype=dtype)
        B = torch.randn(bucket_K, N, dtype=dtype)
        A = A.to(device)
        B = B.to(device)
        xm.mark_step()
        xm.wait_device_ops()
    
    latencies = []
    for _ in range(iterations):
        A = torch.randn(M, bucket_K, dtype=dtype)
        B = torch.randn(bucket_K, N, dtype=dtype)
        
        start = time.perf_counter()
        A = A.to(device)
        B = B.to(device)
        xm.mark_step()
        xm.wait_device_ops()
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)
    
    return np.mean(latencies), np.std(latencies)


def measure_e2e_with_padding(
    compiled_model,
    M: int, 
    actual_K: int, 
    bucket_K: int, 
    N: int,
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iterations: int = 100
) -> Tuple[float, float]:
    """
    전체 E2E 시간 측정 (vLLM 실제 동작 시뮬레이션)
    
    포함:
    1. CPU에서 텐서 생성
    2. CPU에서 Padding (F.pad)
    3. CPU → NPU 전송
    4. NPU matmul 연산
    5. 동기화
    """
    device = torch.device('xla:0')
    import torch_xla.core.xla_model as xm
    
    # Warmup
    for _ in range(warmup):
        A = torch.randn(M, actual_K, dtype=dtype)
        B = torch.randn(actual_K, N, dtype=dtype)
        
        if actual_K < bucket_K:
            A = F.pad(A, (0, bucket_K - actual_K), value=0.0)
            B = F.pad(B, (0, 0, 0, bucket_K - actual_K), value=0.0)
        
        A = A.to(device)
        B = B.to(device)
        
        results = compiled_model(A, B)
        _ = results.sum()  # 강제로 결과 사용
        xm.mark_step()
        xm.wait_device_ops()
    
    # 측정
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        # 1. CPU에서 텐서 생성
        A = torch.randn(M, actual_K, dtype=dtype)
        B = torch.randn(actual_K, N, dtype=dtype)
        
        # 2. CPU에서 Padding (vLLM 방식)
        if actual_K < bucket_K:
            A = F.pad(A, (0, bucket_K - actual_K), value=0.0)
            B = F.pad(B, (0, 0, 0, bucket_K - actual_K), value=0.0)
        
        # 3. CPU → NPU 전송
        A = A.to(device)
        B = B.to(device)
        
        # 4. NPU 연산 + 동기화
        results = compiled_model(A, B)
        _ = results.sum()  # 강제로 결과 사용
        xm.mark_step()
        xm.wait_device_ops()
        
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    return np.mean(latencies), np.std(latencies)


def measure_e2e_pipeline(
    compiled_models: Dict[int, torch.jit.ScriptModule],
    bucket_manager: BucketManager,
    M: int, K: int, actual_N: int,
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iterations: int = 100
) -> Tuple[float, float]:
    """
    전체 파이프라인 시간 측정 (vLLM 실제 동작 시뮬레이션)
    
    포함되는 시간:
    1. Bucket 선택 (CPU)
    2. 텐서 생성 (CPU)
    3. Padding (CPU)
    4. Device 전송
    5. NPU 연산
    6. 동기화
    """
    device = torch.device('xla:0')
    import torch_xla.core.xla_model as xm

    # Warmup (bucket 선택 + padding + 실행)
    for _ in range(warmup):
        bucket_N = bucket_manager.get_bucket(actual_N)
        compiled_model = compiled_models[bucket_N]
        
        A = torch.randn(M, K, dtype=dtype)
        B = torch.randn(K, actual_N, dtype=dtype)
        
        if actual_N < bucket_N:
            B = F.pad(B, (0, bucket_N - actual_N), value=0.0)
        
        A = A.to(device)
        B = B.to(device)
        
        results = compiled_model(A, B)
        _ = results.sum()  # 강제로 결과 사용
        xm.mark_step()
        xm.wait_device_ops()

    # 측정
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        # === vLLM 파이프라인 시작 ===
        # 1. Bucket 선택 (CPU)
        bucket_N = bucket_manager.get_bucket(actual_N)
        compiled_model = compiled_models[bucket_N]
        
        # 2. 텐서 생성 (CPU)
        A = torch.randn(M, K, dtype=dtype)
        B = torch.randn(K, actual_N, dtype=dtype)
        
        # 3. Padding (CPU)
        if actual_N < bucket_N:
            B = F.pad(B, (0, bucket_N - actual_N), value=0.0)
        
        # 4. Device 전송
        A = A.to(device)
        B = B.to(device)
        
        # 5. NPU 연산 + 동기화
        results = compiled_model(A, B)
        _ = results.sum()  # 강제로 결과 사용
        xm.mark_step()
        xm.wait_device_ops()
        # === 파이프라인 끝 ===
        
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return np.mean(latencies), np.std(latencies)




def run_attention_profile(
    M: int = 128,  # 고정
    N: int = 128,  # 고정
    bucket_sizes: List[int] = [128, 256, 512, 1024, 2048, 4096],
    output_file: str = '../results/matmul/attention_profile.csv'
):
    """
    Attention 연산 프로파일링 (Llama3.1-8B)

    A (M, K) @ B (K, N) = C (M, N)

    M, N = 128 (고정)
    K = seq_len (bucketing)
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    bucket_manager = BucketManager(bucket_sizes)

    print("=" * 70)
    print("Attention 연산 프로파일링 (Llama3.1-8B)")
    print("=" * 70)
    print(f"\n설정:")
    print(f"  M (고정): {M}")
    print(f"  N (고정): {N}")
    print(f"  K (seq_len, bucketing): {bucket_sizes}")
    print()

    # 각 bucket 크기로 컴파일
    print("Step 1: Bucket 모델 컴파일")
    print("-" * 40)
    compiled_models = {}
    model = MatmulModule().eval()

    for bucket_K in bucket_sizes:
        print(f"  [컴파일] Shape: ({M}, {bucket_K}) x ({bucket_K}, {N})", end='', flush=True)

        A = torch.randn(M, bucket_K, dtype=torch.float16)
        B = torch.randn(bucket_K, N, dtype=torch.float16)

        workdir = f"/tmp/neuron_bucket_attention_{M}x{bucket_K}x{N}"
        compiled_model = torch_neuronx.trace(model, (A, B), compiler_workdir=workdir)
        compiled_models[bucket_K] = compiled_model

        print(" ✓")

    # 테스트 케이스 생성
    test_cases = []
    for bucket in bucket_sizes:
        prev_bucket = bucket_sizes[bucket_sizes.index(bucket) - 1] if bucket_sizes.index(bucket) > 0 else 1

        sizes_in_bucket = [
            prev_bucket + 1,
            int(bucket * 0.25),
            int(bucket * 0.5),
            int(bucket * 0.75),
            int(bucket * 0.9),
            bucket - 1,
            bucket,
        ]
        sizes_in_bucket = sorted(set(s for s in sizes_in_bucket if prev_bucket < s <= bucket))

        for size in sizes_in_bucket:
            test_cases.append((size, bucket))

    print("\nStep 2: 측정 시작")
    print("-" * 40)
    print(f"총 테스트 케이스: {len(test_cases)}개\n")

    results = []

    for idx, (actual_K, expected_bucket) in enumerate(test_cases, 1):
        bucket_K = bucket_manager.get_bucket(actual_K)
        assert bucket_K == expected_bucket

        padding_size = bucket_K - actual_K
        padding_pct = (padding_size / actual_K * 100) if actual_K > 0 else 0

        print(f"[{idx:2d}/{len(test_cases)}] seq_len={actual_K:4d} → Bucket={bucket_K:4d} "
              f"(padding: {padding_size:3d}, {padding_pct:5.1f}%)")

        # 1. NPU-only 측정 (padding 제외, 순수 연산)
        npu_mean, npu_std = measure_npu_only(
            compiled_models[bucket_K], M, bucket_K, N
        )
        print(f"       NPU-only:     {npu_mean:.4f} ± {npu_std:.4f} ms")

        # 2. Padding-only 측정 (CPU에서 F.pad 시간)
        pad_mean, pad_std = measure_padding_only(
            M, actual_K, bucket_K, N
        )
        print(f"       Padding-only: {pad_mean:.4f} ± {pad_std:.4f} ms")

        # 3. Transfer-only 측정 (CPU → NPU 전송)
        transfer_mean, transfer_std = measure_transfer_only(
            M, bucket_K, N
        )
        print(f"       Transfer-only:{transfer_mean:.4f} ± {transfer_std:.4f} ms")

        # 4. E2E 전체 측정 (padding + transfer + NPU)
        e2e_mean, e2e_std = measure_e2e_with_padding(
            compiled_models[bucket_K], M, actual_K, bucket_K, N
        )
        print(f"       E2E total:    {e2e_mean:.4f} ± {e2e_std:.4f} ms")
        
        # 오버헤드 계산
        overhead_ms = e2e_mean - npu_mean
        overhead_pct = (overhead_ms / npu_mean * 100) if npu_mean > 0 else 0
        print(f"       Overhead:     {overhead_ms:.4f} ms ({overhead_pct:.1f}%)\n")

        results.append({
            'seq_len_actual': actual_K,
            'seq_len_bucket': bucket_K,
            'M': M,
            'N': N,
            'padding_size': padding_size,
            'padding_pct': padding_pct,
            'npu_only_ms': npu_mean,
            'npu_only_std': npu_std,
            'padding_only_ms': pad_mean,
            'padding_only_std': pad_std,
            'transfer_only_ms': transfer_mean,
            'transfer_only_std': transfer_std,
            'e2e_total_ms': e2e_mean,
            'e2e_total_std': e2e_std,
            'overhead_ms': overhead_ms,
            'overhead_pct': overhead_pct,
        })

    # CSV 저장
    with open(output_file, 'w', newline='') as f:
        fieldnames = [
            'seq_len_actual', 'seq_len_bucket', 'M', 'N',
            'padding_size', 'padding_pct',
            'npu_only_ms', 'npu_only_std',
            'padding_only_ms', 'padding_only_std',
            'transfer_only_ms', 'transfer_only_std',
            'e2e_total_ms', 'e2e_total_std',
            'overhead_ms', 'overhead_pct',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n결과 저장: {output_file}")
    print("=" * 70)


def run_ffn_profile(
    K: int = 4096,  # hidden_dim (고정)
    N: int = 14336,  # intermediate_dim (고정)
    bucket_sizes: List[int] = [128, 256, 512, 1024, 2048, 4096],
    output_file: str = '../results/matmul/ffn_profile.csv'
):
    """
    FFN 연산 프로파일링 (Llama3.1-8B)

    (batch*seq, hidden_dim) @ (hidden_dim, intermediate_dim)

    M = batch*seq (bucketing)
    K = hidden_dim (4096, 고정)
    N = intermediate_dim (14336, 고정)
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    bucket_manager = BucketManager(bucket_sizes)

    print("=" * 70)
    print("FFN 연산 프로파일링 (Llama3.1-8B)")
    print("=" * 70)
    print(f"\n설정:")
    print(f"  M (batch*seq, bucketing): {bucket_sizes}")
    print(f"  K (hidden_dim, 고정): {K}")
    print(f"  N (intermediate_dim, 고정): {N}")
    print()

    # 각 bucket 크기로 컴파일
    print("Step 1: Bucket 모델 컴파일")
    print("-" * 40)
    compiled_models = {}
    model = MatmulModule().eval()

    for bucket_M in bucket_sizes:
        print(f"  [컴파일] Shape: ({bucket_M}, {K}) x ({K}, {N})", end='', flush=True)

        A = torch.randn(bucket_M, K, dtype=torch.float16)
        B = torch.randn(K, N, dtype=torch.float16)

        workdir = f"/tmp/neuron_bucket_ffn_{bucket_M}x{K}x{N}"
        compiled_model = torch_neuronx.trace(model, (A, B), compiler_workdir=workdir)
        compiled_models[bucket_M] = compiled_model

        print(" ✓")

    # 테스트 케이스 생성
    test_cases = []
    for bucket in bucket_sizes:
        prev_bucket = bucket_sizes[bucket_sizes.index(bucket) - 1] if bucket_sizes.index(bucket) > 0 else 1

        sizes_in_bucket = [
            prev_bucket + 1,
            int(bucket * 0.5),
            int(bucket * 0.75),
            int(bucket * 0.9),
            bucket - 1,
            bucket,
        ]
        sizes_in_bucket = sorted(set(s for s in sizes_in_bucket if prev_bucket < s <= bucket))

        for size in sizes_in_bucket:
            test_cases.append((size, bucket))

    print("\nStep 2: 측정 시작")
    print("-" * 40)
    print(f"총 테스트 케이스: {len(test_cases)}개\n")

    results = []

    for idx, (actual_M, expected_bucket) in enumerate(test_cases, 1):
        bucket_M = bucket_manager.get_bucket(actual_M)
        assert bucket_M == expected_bucket

        padding_size = bucket_M - actual_M
        padding_pct = (padding_size / actual_M * 100) if actual_M > 0 else 0

        print(f"[{idx:2d}/{len(test_cases)}] batch*seq={actual_M:4d} → Bucket={bucket_M:4d} "
              f"(padding: {padding_size:3d}, {padding_pct:5.1f}%)")

        # NPU-only 측정
        npu_mean, npu_std = measure_npu_only(
            compiled_models[bucket_M], bucket_M, K, N
        )

        results.append({
            'batch_seq_actual': actual_M,
            'batch_seq_bucket': bucket_M,
            'hidden_dim': K,
            'intermediate_dim': N,
            'padding_size': padding_size,
            'padding_pct': padding_pct,
            'npu_only_ms': npu_mean,
            'npu_only_std': npu_std,
        })

        print(f"       NPU-only: {npu_mean:.4f} ± {npu_std:.4f} ms\n")

    # CSV 저장
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['batch_seq_actual', 'batch_seq_bucket', 'hidden_dim', 'intermediate_dim',
                     'padding_size', 'padding_pct', 'npu_only_ms', 'npu_only_std']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n결과 저장: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    print("=" * 70)
    print("LLM Inference Profiler (Llama3.1-8B)")
    print("=" * 70)
    print()

    # # Attention 연산 프로파일링
    # run_attention_profile(
    #     M=128,  # 고정
    #     N=128,  # 고정
    #     bucket_sizes=[128, 256, 512, 1024, 2048, 4096],  # K bucketing
    #     output_file='../results/matmul/attention_profile.csv'
    # )


    #FFN 연산 프로파일링 (주석 처리 - 필요 시 활성화)
    run_ffn_profile(
        K=4096,  # hidden_dim
        N=14336,  # intermediate_dim
        bucket_sizes=[128, 256, 512, 1024, 2048, 4096],
        output_file='../results/matmul/ffn_profile.csv'
    )

    print("\n" + "=" * 70 + "\n")