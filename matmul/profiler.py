"""
Simple NPU Matmul Profiler
A(M×K) × B(K×N) = C(M×N) latency 측정
"""

import torch
import torch_neuronx
import time
import csv
import os


class CycleCalculator:
    """
    Matmul cycle 계산
    cycle = M + N + K - 2
    """

    @staticmethod
    def calculate_cycles(M, K, N):
        """
        행렬곱 cycle 계산

        Args:
            M, K, N: 행렬 차원

        Returns:
            cycle 수
        """
        return M + N + K - 2

    @staticmethod
    def latency_per_cycle(latency_ms, M, K, N):
        """
        Cycle당 latency 계산

        Args:
            latency_ms: 전체 latency (ms)
            M, K, N: 행렬 차원

        Returns:
            cycle당 latency (ms/cycle)
        """
        cycles = CycleCalculator.calculate_cycles(M, K, N)
        return latency_ms / cycles if cycles > 0 else 0


def measure_matmul(M, K, N, dtype=torch.float16, warmup=3, iterations=20):
    """
    행렬곱 latency 측정

    Args:
        M, K, N: 행렬 차원
        dtype: 데이터 타입 (default: fp16)
        warmup: warmup 반복 횟수
        iterations: 측정 반복 횟수

    Returns:
        평균 latency (ms)
    """
    # Neuron device
    device = torch.device('xla:0')

    # 텐서 생성 (CPU)
    A = torch.randn(M, K, dtype=dtype)
    B = torch.randn(K, N, dtype=dtype)

    # Matmul 모델
    class MatmulModule(torch.nn.Module):
        def forward(self, a, b):
            return torch.matmul(a, b)

    model = MatmulModule().eval()

    # Neuron 컴파일
    print(f"  컴파일 중... ", end='', flush=True)
    compiled_model = torch_neuronx.trace(
        model,
        (A, B),
        compiler_workdir=f"/tmp/neuron_matmul_{M}x{K}x{N}"
    )
    print("완료")

    # Device로 이동
    A = A.to(device)
    B = B.to(device)

    # XLA 동기화 함수
    import torch_xla.core.xla_model as xm

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

    return sum(latencies) / len(latencies)


def run_profiling(M_values, K_values, N_values, output_file='../results/matmul/test.csv'):
    """
    M, K, N 조합에 대해 프로파일링 실행

    Args:
        M_values, K_values, N_values: 테스트할 값들 (리스트)
        output_file: 결과 저장 경로
    """
    # 결과 저장 디렉토리 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = []
    count = 0

    print(f"프로파일링 시작 (N 기준, M < N and M <= 128)\n")

    # N을 기준으로 순회
    for N in N_values:
        for K in K_values:
            for M in M_values:
                # M은 N보다 작아야 하고, 128 이하여야 함
                if M >= N:
                    continue
                if M > 128:
                    break  # M_values가 오름차순이므로 break

                count += 1
                print(f"[{count}] M={M}, K={K}, N={N}")

                try:
                    latency = measure_matmul(M, K, N)
                    cycles = CycleCalculator.calculate_cycles(M, K, N)
                    lat_per_cycle = CycleCalculator.latency_per_cycle(latency, M, K, N)

                    results.append({
                        'M': M,
                        'K': K,
                        'N': N,
                        'latency_ms': latency,
                        'cycles': cycles,
                        'latency_per_cycle': lat_per_cycle
                    })
                    print(f"  Latency: {latency:.4f} ms, Cycles: {cycles}, Lat/Cycle: {lat_per_cycle:.6f} ms\n")

                except Exception as e:
                    print(f"  오류: {e}\n")
                    continue

    # CSV 저장
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['M', 'K', 'N', 'latency_ms', 'cycles', 'latency_per_cycle'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n완료! 결과 저장: {output_file}")
    print(f"측정 성공: {len(results)}개")


if __name__ == '__main__':
    # 테스트 범위 설정
    # M: 128부터 512까지 128 간격
    M_start, M_end, M_step = 32, 128, 32
    M_values = range(M_start, M_end + 1, M_step)

    # K: 128부터 256까지 128 간격
    K_start, K_end, K_step = 32, 128, 32
    K_values = range(K_start, K_end + 1, K_step)

    # N: 512부터 1024까지 512 간격
    N_start, N_end, N_step = 32, 512, 32
    N_values = range(N_start, N_end + 1, N_step)

    run_profiling(M_values, K_values, N_values, '../results/matmul/simple_test.csv')
