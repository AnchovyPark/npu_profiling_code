"""
Custom NPU 프로파일링 예제
원하는 행렬 크기를 직접 지정해서 측정
"""

from npu_matmul_profiler import NPUMatmulProfiler, MatmulConfig

# Profiler 초기화
profiler = NPUMatmulProfiler(use_neuron=True)

# 측정하고 싶은 행렬 크기 정의
# M=128, K=128 고정, N을 128부터 1024까지 32씩 증가
configs = []
M = 129
K = 128
for N in range(128, 2048 + 1, 32):
    configs.append(MatmulConfig(M=M, K=K, N=N))

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
