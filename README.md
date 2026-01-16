# NPU Matrix Multiplication Profiling

NPU의 PE(128x128) 및 moving tensor(128x512) 제약을 고려한 행렬곱 latency profiling 도구입니다.

## NPU 제약사항

- **PE (Processing Element)**: 128 x 128
- **Moving Tensor**: 128 x 512 (TensorE)

## 주요 기능

1. **Tiling 계산**: 다양한 행렬 크기에 대해 필요한 tile 개수 자동 계산
2. **Latency 측정**: 반복 실행을 통한 정확한 latency 측정
3. **다양한 프로파일링 모드**:
   - `llm`: LLM inference를 위한 typical dimensions (prefill/decode)
   - `tiling`: Tiling boundary 중심 테스트
   - `sweep`: 전체 dimension 범위 스윕

## 사용법

### 기본 실행

```bash
python npu_matmul_profiler.py
```

### Custom 프로파일링

```python
from npu_matmul_profiler import NPUMatmulProfiler, MatmulConfig

profiler = NPUMatmulProfiler()

# 단일 행렬곱 프로파일링
config = MatmulConfig(M=1024, K=4096, N=4096)
result = profiler.profile_single_matmul(config, warmup=5, iterations=20)
print(f"Latency: {result.latency_ms:.4f} ms")
print(f"Total tiles: {result.total_tiles}")

# Custom configurations
configs = [
    MatmulConfig(M=256, K=1024, N=1024),
    MatmulConfig(M=512, K=2048, N=2048),
]
results = [profiler.profile_single_matmul(cfg) for cfg in configs]
profiler.save_results_to_csv(results, "custom_results.csv")
```

## NPU API 통합

현재 코드는 NumPy를 placeholder로 사용합니다. 실제 NPU를 사용하려면 `run_npu_matmul` 메소드를 수정하세요:

```python
def run_npu_matmul(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    start_time = time.perf_counter()

    # 실제 NPU API 호출로 교체
    # 예시:
    # - PyTorch NPU: torch_npu.matmul(A, B)
    # - TensorFlow Lite: interpreter.invoke()
    # - Custom NPU runtime

    C = your_npu_api.matmul(A, B)

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    return C, latency_ms
```

## 출력 형식

결과는 CSV 파일로 저장되며, 다음 정보를 포함합니다:

- `M`, `K`, `N`: 행렬 차원
- `latency_ms`: 측정된 latency (밀리초)
- `num_tiles_M`, `num_tiles_K`, `num_tiles_N`: 각 차원의 tile 개수
- `total_tiles`: 전체 tile 개수

## LLM Inference 고려사항

### Prefill Phase
- Batch size × Sequence length가 M 차원
- 다양한 sequence length 테스트 (1, 64, 128, ..., 2048)

### Decode Phase
- Batch size만 M 차원 (sequence length = 1)
- 높은 batch size에서의 성능 측정

### FFN Layers
- Hidden dim → Intermediate dim (4x)
- Intermediate dim → Hidden dim

## vLLM Bucketing 참고

vLLM은 다음과 같이 bucketing을 수행합니다:
- Sequence length bucketing: [128, 256, 512, 1024, 2048]
- Batch size bucketing: power of 2

이를 참고하여 profiling 결과를 분석하고 최적의 bucket을 설정할 수 있습니다.
