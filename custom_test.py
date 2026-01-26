"""
Custom NPU 프로파일링 예제
원하는 행렬 크기를 직접 지정해서 측정
"""

from npu_matmul_profiler import NPUMatmulProfiler, MatmulConfig

# Profiler 초기화
profiler = NPUMatmulProfiler(use_neuron=True)

# 측정하고 싶은 행렬 크기 정의
# PE와 Moving Tensor Boundary 테스트 (K 변화)
configs = []

# PE_K의 배수 (128의 배수)
K_values = [128, 256, 384, 512]

# PE_M의 배수 (128의 배수)
M_values = [512, 640, 768, 896, 1024]

# MOVING_N의 배수 (512의 배수)
N_values = [1024, 1536, 2048, 2560, 3072, 3584, 4096]

for K in K_values:
    for M in M_values:
        for N in N_values:
            configs.append(MatmulConfig(M=M, K=K, N=N))

results = []

for config in configs:
    print(f'\n프로파일링: {config}')

    # Tiling 계산
    tiles_M, tiles_K, tiles_N = profiler.calculate_tiling(config)
    print(f'  Tiling: M={tiles_M} x K={tiles_K} x N={tiles_N} = {tiles_M*tiles_K*tiles_N} tiles')

    # 실제 측정 (warmup=3, iterations=10)
    result = profiler.profile_single_matmul(config, warmup=3, iterations=20)

    print(f'  Latency: {result.latency_ms:.4f} ms')
    results.append(result)

# CSV로 저장
profiler.save_results_to_csv(results, './results/K_variation_test.csv')
print('\n결과가 ./results/K_variation_test.csv에 저장되었습니다.')
