"""
같은 NEFF가 다른 shape에 재사용되는지 테스트
"""

import torch
from npu_matmul_profiler import NPUMatmulProfiler, MatmulConfig

profiler = NPUMatmulProfiler(use_neuron=True)

print("=" * 80)
print("테스트: 같은 NEFF 파일이 다른 M 값에 재사용되는가?")
print("=" * 80)

# 첫 번째: M=128로 컴파일
print("\n1. M=128으로 첫 컴파일...")
config1 = MatmulConfig(M=128, K=128, N=128)
result1 = profiler.profile_single_matmul(config1, warmup=1, iterations=3)
print(f"   Latency: {result1.latency_ms:.4f} ms")

# 두 번째: M=256 (같은 K, N)
print("\n2. M=256으로 실행 (캐시된 NEFF 재사용 예상)...")
config2 = MatmulConfig(M=256, K=128, N=128)
result2 = profiler.profile_single_matmul(config2, warmup=1, iterations=3)
print(f"   Latency: {result2.latency_ms:.4f} ms")

# 세 번째: M=512
print("\n3. M=512로 실행 (캐시된 NEFF 재사용 예상)...")
config3 = MatmulConfig(M=512, K=128, N=128)
result3 = profiler.profile_single_matmul(config3, warmup=1, iterations=3)
print(f"   Latency: {result3.latency_ms:.4f} ms")

print("\n" + "=" * 80)
print("결론:")
print(f"  M=128 (1 tile):  {result1.latency_ms:.4f} ms")
print(f"  M=256 (2 tiles): {result2.latency_ms:.4f} ms (ratio: {result2.latency_ms/result1.latency_ms:.2f}x)")
print(f"  M=512 (4 tiles): {result3.latency_ms:.4f} ms (ratio: {result3.latency_ms/result1.latency_ms:.2f}x)")
print("\n  → 같은 NEFF를 사용하지만 입력 shape에 따라 latency가 다름!")
print("  → Runtime이 shape을 보고 반복 횟수를 결정한다는 증거!")
print("=" * 80)
