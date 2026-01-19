"""
Custom NPU 프로파일링 예제
원하는 행렬 크기를 직접 지정해서 측정
"""

from npu_matmul_profiler import NPUMatmulProfiler, MatmulConfig

# Profiler 초기화
profiler = NPUMatmulProfiler(use_neuron=True)

# 측정하고 싶은 행렬 크기 정의
configs = [
    MatmulConfig(M=128, K=4096, N=4096),   # 예: Prefill
    MatmulConfig(M=256, K=4096, N=4096),   # 예: Larger prefill
    MatmulConfig(M=16, K=4096, N=4096),    # 예: Decode with batch
    # 여기에 원하는 크기 추가
]

results = []

for config in configs:
    print(f'\n프로파일링: {config}')

    # Tiling 계산
    tiles_M, tiles_K, tiles_N = profiler.calculate_tiling(config)
    print(f'  Tiling: M={tiles_M} x K={tiles_K} x N={tiles_N} = {tiles_M*tiles_K*tiles_N} tiles')

    # 실제 측정 (warmup=3, iterations=10)
    result = profiler.profile_single_matmul(config, warmup=3, iterations=10)

    print(f'  Latency: {result.latency_ms:.4f} ms')
    results.append(result)

# CSV로 저장
profiler.save_results_to_csv(results, './results/custom_test.csv')
print('\n결과가 ./results/custom_test.csv에 저장되었습니다.')
