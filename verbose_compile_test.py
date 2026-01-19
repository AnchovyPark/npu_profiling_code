"""
Verbose compilation test - 컴파일 과정을 자세히 볼 수 있는 테스트
"""

import os
# Enable verbose Neuron compiler output
os.environ['NEURON_CC_FLAGS'] = '--verbose info'
os.environ['NEURONX_DUMP_TO'] = '/tmp/neuron_debug'

from npu_matmul_profiler import NPUMatmulProfiler, MatmulConfig

# Profiler 초기화
profiler = NPUMatmulProfiler(use_neuron=True)

# 간단한 테스트 케이스 - boundary 비교
configs = [
    MatmulConfig(M=128, K=128, N=128),   # Exact boundary (1 tile)
    MatmulConfig(M=129, K=128, N=128),   # Just over boundary (2 tiles)
    MatmulConfig(M=256, K=128, N=128),   # 2x boundary (2 tiles)
]

print("=" * 80)
print("Verbose Compilation Test")
print("Neuron compiler output will show detailed tiling information")
print("=" * 80)

for config in configs:
    print(f'\n{"=" * 80}')
    print(f'Testing: {config}')
    print(f'{"=" * 80}')

    # Tiling 계산
    tiles_M, tiles_K, tiles_N = profiler.calculate_tiling(config)
    print(f'Calculated Tiling: M={tiles_M} x K={tiles_K} x N={tiles_N} = {tiles_M*tiles_K*tiles_N} tiles')

    # 컴파일 & 측정 (첫 번째는 컴파일 시간 포함)
    result = profiler.profile_single_matmul(config, warmup=1, iterations=5)

    print(f'Measured Latency: {result.latency_ms:.4f} ms')

print(f'\n{"=" * 80}')
print("Check /tmp/neuron_debug/ for detailed compiler artifacts")
print(f'{"=" * 80}')
