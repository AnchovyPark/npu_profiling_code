"""
Bucketing-based NPU Matmul Profiler
vLLM-Neuron의 bucketing 방식을 사용한 matmul 프로파일링
특정 bucket 크기로만 compile하고, 실제 입력은 padding하여 실행
"""

import torch
import torch_neuronx
import time
import csv
import os


class BucketManager:
    """Bucket 관리 및 선택 로직"""

    # vLLM-Neuron에서 일반적으로 사용하는 bucketing 크기
    # sequence length bucketing 패턴을 matmul의 N 차원에 적용
    DEFAULT_BUCKETS = [128, 256, 512, 1024, 2048, 4096]

    def __init__(self, buckets=None):
        """
        Args:
            buckets: bucket 크기 리스트. None이면 DEFAULT_BUCKETS 사용
        """
        self.buckets = sorted(buckets if buckets else self.DEFAULT_BUCKETS)

    def get_bucket(self, size):
        """
        주어진 크기에 대해 가장 가까운 bucket 반환

        Args:
            size: 입력 크기

        Returns:
            가장 가까운 큰 bucket 크기. 없으면 가장 큰 bucket
        """
        for bucket in self.buckets:
            if size <= bucket:
                return bucket
        # 모든 bucket보다 크면 가장 큰 bucket 반환
        return self.buckets[-1]

    def needs_padding(self, size):
        """크기가 padding이 필요한지 확인"""
        bucket = self.get_bucket(size)
        return size < bucket


class BucketedMatmulModule(torch.nn.Module):
    """Bucketing을 지원하는 Matmul 모듈"""

    def __init__(self, bucket_size):
        super().__init__()
        self.bucket_size = bucket_size

    def forward(self, a, b):
        """
        Matmul 수행. 입력이 bucket보다 작으면 padding됨

        Args:
            a: (M, K) 텐서
            b: (K, N) 텐서 - N은 bucket_size와 같거나 작아야 함
        """
        return torch.matmul(a, b)


def compile_bucket_models(M, K, bucket_manager, dtype=torch.float16, workdir_base="/tmp/neuron_matmul_bucketed"):
    """
    각 bucket 크기에 대해 모델 컴파일

    Args:
        M, K: 행렬 차원 (고정)
        bucket_manager: BucketManager 인스턴스
        dtype: 데이터 타입
        workdir_base: 컴파일 작업 디렉토리 베이스

    Returns:
        {bucket_size: compiled_model} 딕셔너리
    """
    compiled_models = {}

    for bucket_N in bucket_manager.buckets:
        print(f"\n[컴파일] M={M}, K={K}, N(bucket)={bucket_N}")

        # 텐서 생성
        A = torch.randn(M, K, dtype=dtype)
        B = torch.randn(K, bucket_N, dtype=dtype)

        # 모델 생성
        model = BucketedMatmulModule(bucket_N).eval()

        # 컴파일
        workdir = f"{workdir_base}_{M}x{K}x{bucket_N}"
        print(f"  컴파일 중... (workdir: {workdir})", end='', flush=True)

        compiled_model = torch_neuronx.trace(
            model,
            (A, B),
            compiler_workdir=workdir
        )

        compiled_models[bucket_N] = compiled_model
        print(" 완료")

    return compiled_models


def measure_bucketed_matmul(M, K, N, bucket_manager, compiled_models,
                           dtype=torch.float16, warmup=3, iterations=20):
    """
    Bucketing 방식으로 matmul latency 측정

    Args:
        M, K, N: 실제 행렬 차원
        bucket_manager: BucketManager 인스턴스
        compiled_models: {bucket_size: compiled_model} 딕셔너리
        dtype: 데이터 타입
        warmup: warmup 반복 횟수
        iterations: 측정 반복 횟수

    Returns:
        (평균 latency (ms), 사용된 bucket 크기)
    """
    device = torch.device('xla:0')
    import torch_xla.core.xla_model as xm

    # 적절한 bucket 선택
    bucket_N = bucket_manager.get_bucket(N)
    compiled_model = compiled_models[bucket_N]

    # 텐서 생성 (실제 크기)
    A = torch.randn(M, K, dtype=dtype)
    B_actual = torch.randn(K, N, dtype=dtype)

    # N을 bucket 크기로 padding
    if N < bucket_N:
        # Zero padding
        pad_size = bucket_N - N
        B = torch.nn.functional.pad(B_actual, (0, pad_size, 0, 0), value=0.0)
    else:
        B = B_actual

    # Device로 이동
    A = A.to(device)
    B = B.to(device)

    # Warmup
    for _ in range(warmup):
        C = compiled_model(A, B)
        xm.mark_step()
        xm.wait_device_ops()

    # 측정
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = compiled_model(A, B)
        xm.mark_step()
        xm.wait_device_ops()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    avg_latency = sum(latencies) / len(latencies)
    return avg_latency, bucket_N


def run_bucketed_profiling(M, K, test_N_values, buckets=None,
                          output_file='../results/matmul/bucketed_profile.csv'):
    """
    Bucketing 방식으로 프로파일링 실행

    Args:
        M, K: 고정 행렬 차원
        test_N_values: 테스트할 N 값들
        buckets: bucket 크기들 (None이면 DEFAULT_BUCKETS 사용)
        output_file: 결과 저장 경로
    """
    # 결과 저장 디렉토리 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Bucket 관리자 생성
    bucket_manager = BucketManager(buckets)
    print(f"Bucket 크기: {bucket_manager.buckets}")

    # Step 1: 각 bucket 크기로 모델 컴파일
    print(f"\n{'='*60}")
    print(f"Step 1: Bucket 모델 컴파일 (M={M}, K={K})")
    print(f"{'='*60}")
    compiled_models = compile_bucket_models(M, K, bucket_manager)

    # Step 2: 다양한 N 크기로 프로파일링
    print(f"\n{'='*60}")
    print(f"Step 2: 프로파일링 시작")
    print(f"{'='*60}\n")

    results = []

    for idx, N in enumerate(test_N_values, 1):
        bucket_N = bucket_manager.get_bucket(N)
        padding_needed = N < bucket_N

        print(f"[{idx}/{len(test_N_values)}] M={M}, K={K}, N={N} -> Bucket={bucket_N} "
              f"(padding={'Yes' if padding_needed else 'No'})")

        try:
            latency, used_bucket = measure_bucketed_matmul(
                M, K, N, bucket_manager, compiled_models
            )

            # Padding overhead 계산
            padding_ratio = (used_bucket - N) / N if N > 0 else 0

            results.append({
                'M': M,
                'K': K,
                'N_actual': N,
                'N_bucket': used_bucket,
                'padding_size': used_bucket - N,
                'padding_ratio': padding_ratio,
                'latency_ms': latency,
            })

            print(f"  Latency: {latency:.4f} ms, "
                  f"Padding: {used_bucket - N} ({padding_ratio*100:.1f}%)\n")

        except Exception as e:
            print(f"  오류: {e}\n")
            continue

    # CSV 저장
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['M', 'K', 'N_actual', 'N_bucket', 'padding_size',
                     'padding_ratio', 'latency_ms']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"완료! 결과 저장: {output_file}")
    print(f"측정 성공: {len(results)}/{len(test_N_values)}개")
    print(f"{'='*60}")

    # 통계 출력
    if results:
        avg_latency = sum(r['latency_ms'] for r in results) / len(results)
        avg_padding_ratio = sum(r['padding_ratio'] for r in results) / len(results)
        print(f"\n통계:")
        print(f"  평균 Latency: {avg_latency:.4f} ms")
        print(f"  평균 Padding 비율: {avg_padding_ratio*100:.2f}%")
        print(f"  사용된 Bucket 개수: {len(set(r['N_bucket'] for r in results))}/{len(bucket_manager.buckets)}")


def compile_and_measure_experiments(experiments, buckets_config, output_file):
    """
    다양한 (M, K, N) 조합에 대해 bucketing 기반 프로파일링 수행

    Args:
        experiments: [(M, K, N), ...] 리스트
        buckets_config: {'M': [...], 'K': [...], 'N': [...]} bucket 설정
        output_file: 결과 저장 경로
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Bucket 관리자들 생성
    bucket_managers = {
        'M': BucketManager(buckets_config.get('M')),
        'K': BucketManager(buckets_config.get('K')),
        'N': BucketManager(buckets_config.get('N'))
    }

    print(f"\n{'='*60}")
    print(f"Bucketing 설정:")
    for dim, manager in bucket_managers.items():
        print(f"  {dim}: {manager.buckets}")
    print(f"{'='*60}\n")

    # 필요한 bucket 조합 찾기
    unique_buckets = set()
    for M, K, N in experiments:
        bucket_M = bucket_managers['M'].get_bucket(M)
        bucket_K = bucket_managers['K'].get_bucket(K)
        bucket_N = bucket_managers['N'].get_bucket(N)
        unique_buckets.add((bucket_M, bucket_K, bucket_N))

    print(f"컴파일 필요한 bucket 조합: {len(unique_buckets)}개")
    for bucket_combo in sorted(unique_buckets):
        print(f"  {bucket_combo[0]} x {bucket_combo[1]} x {bucket_combo[2]}")
    print()

    # 각 bucket 조합에 대해 컴파일
    print(f"{'='*60}")
    print(f"Step 1: Bucket 모델 컴파일")
    print(f"{'='*60}\n")

    compiled_models = {}
    for idx, (bucket_M, bucket_K, bucket_N) in enumerate(sorted(unique_buckets), 1):
        print(f"[{idx}/{len(unique_buckets)}] M={bucket_M}, K={bucket_K}, N={bucket_N}")

        A = torch.randn(bucket_M, bucket_K, dtype=torch.float16)
        B = torch.randn(bucket_K, bucket_N, dtype=torch.float16)

        model = BucketedMatmulModule(bucket_N).eval()
        workdir = f"/tmp/neuron_matmul_bucketed_{bucket_M}x{bucket_K}x{bucket_N}"

        print(f"  컴파일 중... ", end='', flush=True)
        compiled_model = torch_neuronx.trace(model, (A, B), compiler_workdir=workdir)
        compiled_models[(bucket_M, bucket_K, bucket_N)] = compiled_model
        print("완료")

    # Step 2: 실험 수행
    print(f"\n{'='*60}")
    print(f"Step 2: 프로파일링 시작 ({len(experiments)}개 실험)")
    print(f"{'='*60}\n")

    device = torch.device('xla:0')
    import torch_xla.core.xla_model as xm

    results = []
    for idx, (M, K, N) in enumerate(experiments, 1):
        # 적절한 bucket 선택
        bucket_M = bucket_managers['M'].get_bucket(M)
        bucket_K = bucket_managers['K'].get_bucket(K)
        bucket_N = bucket_managers['N'].get_bucket(N)

        print(f"[{idx}/{len(experiments)}] M={M}, K={K}, N={N} -> "
              f"Bucket=({bucket_M}, {bucket_K}, {bucket_N})")

        compiled_model = compiled_models[(bucket_M, bucket_K, bucket_N)]

        # Bucket 크기로 텐서 생성 (padding 오버헤드 제거)
        # 실제 vLLM에서는 미리 padding된 텐서를 사용
        A = torch.randn(bucket_M, bucket_K, dtype=torch.float16)
        B = torch.randn(bucket_K, bucket_N, dtype=torch.float16)

        # 실제 데이터가 차지하는 영역만 랜덤 값으로 채우고, 나머지는 0
        if M < bucket_M or K < bucket_K:
            A_actual = torch.randn(M, K, dtype=torch.float16)
            A = torch.zeros(bucket_M, bucket_K, dtype=torch.float16)
            A[:M, :K] = A_actual

        if K < bucket_K or N < bucket_N:
            B_actual = torch.randn(K, N, dtype=torch.float16)
            B = torch.zeros(bucket_K, bucket_N, dtype=torch.float16)
            B[:K, :N] = B_actual

        # Device로 이동 (padding은 이미 완료됨)
        A = A.to(device)
        B = B.to(device)

        # Warmup
        for _ in range(5):
            C = compiled_model(A, B)
            xm.mark_step()
            xm.wait_device_ops()

        # 측정 (동기화 오버헤드 최소화를 위해 mark_step만 사용)
        latencies = []
        iterations = 20

        # 전체 배치 실행 후 한 번에 동기화
        xm.mark_step()
        xm.wait_device_ops()

        for _ in range(iterations):
            start = time.perf_counter()
            C = compiled_model(A, B)
            xm.mark_step()
            # Note: 순수 device 실행 시간 측정을 위해 wait는 측정 후에 수행
            xm.wait_device_ops()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        # Padding 정보
        padding_M = bucket_M - M
        padding_K = bucket_K - K
        padding_N = bucket_N - N
        total_padding_ratio = ((bucket_M * bucket_K + bucket_K * bucket_N) - (M * K + K * N)) / (M * K + K * N)

        results.append({
            'M_actual': M,
            'K_actual': K,
            'N_actual': N,
            'M_bucket': bucket_M,
            'K_bucket': bucket_K,
            'N_bucket': bucket_N,
            'padding_M': padding_M,
            'padding_K': padding_K,
            'padding_N': padding_N,
            'padding_ratio': total_padding_ratio,
            'latency_ms': avg_latency,
        })

        print(f"  Latency: {avg_latency:.4f} ms, "
              f"Padding: M+{padding_M}, K+{padding_K}, N+{padding_N} "
              f"({total_padding_ratio*100:.1f}%)\n")

    # CSV 저장
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['M_actual', 'K_actual', 'N_actual',
                     'M_bucket', 'K_bucket', 'N_bucket',
                     'padding_M', 'padding_K', 'padding_N',
                     'padding_ratio', 'latency_ms']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"완료! 결과 저장: {output_file}")
    print(f"측정 성공: {len(results)}/{len(experiments)}개")
    print(f"{'='*60}")

    # 통계 출력
    if results:
        avg_latency = sum(r['latency_ms'] for r in results) / len(results)
        avg_padding_ratio = sum(r['padding_ratio'] for r in results) / len(results)
        print(f"\n통계:")
        print(f"  평균 Latency: {avg_latency:.4f} ms")
        print(f"  평균 Padding 비율: {avg_padding_ratio*100:.2f}%")


if __name__ == '__main__':
    print(f"LLM 추론 연산 프로파일링 (Bucketing 기반)\n")
    print("=" * 60)
    print("주요 개선 사항:")
    print("1. Padding을 미리 수행하여 측정 시 오버헤드 제거")
    print("2. K 차원 고정 (hidden_dim/head_dim)")
    print("3. M, N 차원만 bucketing (sequence length)")
    print("=" * 60)
    print()

    # ========================================
    # Attention 연산 시뮬레이션
    # ========================================
    print("=" * 60)
    print("Attention 연산 프로파일링")
    print("=" * 60)

    # 실제 Attention 연산 시뮬레이션
    # Q @ K^T: (batch*num_heads, seq_q, head_dim) @ (batch*num_heads, head_dim, seq_k)
    #         = (batch*num_heads, seq_q, seq_k)
    # 여기서 bucketing되는 것은 seq_q, seq_k (M, N)
    # head_dim (K)은 고정
    experiments_attention = [
        (128, 128, 128),    # seq_len=128, head_dim=128 (고정)
        (256, 128, 256),    # seq_len=256, head_dim=128 (고정)
        (512, 128, 512),    # seq_len=512, head_dim=128 (고정)
        (1024, 128, 1024),  # seq_len=1024, head_dim=128 (고정)
        (2048, 128, 2048),  # seq_len=2048, head_dim=128 (고정)
        (4096, 128, 4096),  # seq_len=4096, head_dim=128 (고정)
    ]

    buckets_attention = {
        'M': [128, 256, 512, 1024, 2048, 4096],  # seq_len buckets
        'K': [128],  # head_dim 고정
        'N': [128, 256, 512, 1024, 2048, 4096],  # seq_len buckets
    }

    compile_and_measure_experiments(
        experiments=experiments_attention,
        buckets_config=buckets_attention,
        output_file='../results/matmul/attention_bucketed_profile.csv'
    )

    # ========================================
    # FFN 연산 시뮬레이션 (주석 처리)
    # ========================================
    # print("\n" + "=" * 60)
    # print("FFN 연산 프로파일링")
    # print("=" * 60)
    #
    # # FFN 연산 시뮬레이션 (Llama 8B 기준)
    # # (batch*seq, hidden_dim) @ (hidden_dim, intermediate_dim)
    # # 여기서 bucketing되는 것은 batch*seq (M)
    # # hidden_dim, intermediate_dim (K, N)은 고정
    # experiments_ffn = [
    #     (128, 4096, 14336),
    #     (256, 4096, 14336),
    #     (512, 4096, 14336),
    #     (1024, 4096, 14336),
    #     (2048, 4096, 14336),
    # ]
    #
    # buckets_ffn = {
    #     'M': [128, 256, 512, 1024, 2048, 4096],  # batch*seq buckets
    #     'K': [4096],   # hidden_dim 고정
    #     'N': [14336],  # intermediate_dim 고정
    # }
    #
    # compile_and_measure_experiments(
    #     experiments=experiments_ffn,
    #     buckets_config=buckets_ffn,
    #     output_file='../results/matmul/ffn_bucketed_profile.csv'
    # )
