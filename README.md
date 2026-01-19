# NPU Matrix Multiplication Profiling (AWS Neuron)

AWS Inferentia2/Trainium NPU에서 행렬곱 latency를 측정하는 프로파일링 도구입니다.
NPU의 PE(128x128) 및 moving tensor(128x512) 제약을 고려하여 다양한 행렬 크기의 성능을 측정합니다.

## NPU 제약사항

- **PE (Processing Element)**: 128 x 128
- **Moving Tensor**: 128 x 512 (TensorE)
- **컴파일 방식**: Neuron Compiler를 통한 그래프 컴파일 (JIT)

## 요구사항

- **인스턴스**: AWS Inf2 (Inferentia2) 또는 Trn1 (Trainium)
- **OS**: Ubuntu 22.04 또는 Amazon Linux 2
- **Python**: 3.8+

## 설치 방법

### 1. Neuron Driver 및 Runtime 설치

```bash
# APT repository 추가
curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo gpg --dearmor -o /usr/share/keyrings/amazon-neuron-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/amazon-neuron-archive-keyring.gpg] https://apt.repos.neuron.amazonaws.com jammy main" | sudo tee /etc/apt/sources.list.d/neuron.list

# 패키지 설치
sudo apt-get update
sudo apt-get install -y aws-neuronx-runtime-lib aws-neuronx-collectives
```

### 2. PyTorch 및 Neuron SDK 설치

```bash
# PyPI에 Neuron repository 추가
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# PyTorch 및 Neuron SDK 설치
pip install torch==2.9.1 torchvision
pip install neuronx-cc torch-neuronx
```

### 3. 프로젝트 Dependencies 설치

```bash
pip install numpy
```

## 사용법

### 기본 실행

전체 프로파일링 스위트 실행:

```bash
python npu_matmul_profiler.py
```

결과는 `./results/` 디렉토리에 CSV 파일로 저장됩니다.

### 예제 실행

다양한 시나리오별 예제:

```bash
# 모든 예제 실행
python example_usage.py

# 특정 예제만 실행
python example_usage.py 1  # Single matmul
python example_usage.py 2  # Custom configs
python example_usage.py 3  # Tiling boundary
python example_usage.py 4  # LLM scenarios
python example_usage.py 5  # Quick test
```

### Custom 프로파일링 코드 예제

```python
from npu_matmul_profiler import NPUMatmulProfiler, MatmulConfig

# Neuron device를 사용하여 profiler 초기화
profiler = NPUMatmulProfiler(use_neuron=True)

# 단일 행렬곱 프로파일링
config = MatmulConfig(M=1024, K=4096, N=4096)
result = profiler.profile_single_matmul(config, warmup=5, iterations=20)

print(f"Latency: {result.latency_ms:.4f} ms")
print(f"Total tiles: {result.total_tiles}")
print(f"Tiles breakdown: M={result.num_tiles_M}, K={result.num_tiles_K}, N={result.num_tiles_N}")

# 여러 설정 프로파일링
configs = [
    MatmulConfig(M=256, K=1024, N=1024),
    MatmulConfig(M=512, K=2048, N=2048),
]
results = [profiler.profile_single_matmul(cfg) for cfg in configs]
profiler.save_results_to_csv(results, "./results/custom_results.csv")
```

### CPU 모드로 테스트 (Neuron 없이)

Neuron 하드웨어가 없는 환경에서 코드 테스트:

```python
# CPU 모드로 초기화
profiler = NPUMatmulProfiler(use_neuron=False)

# 나머지 사용법은 동일
config = MatmulConfig(M=256, K=512, N=512)
result = profiler.profile_single_matmul(config)
```

## 주요 기능

### 1. 자동 Neuron 컴파일

- 각 행렬 크기별로 자동으로 Neuron 컴파일 수행
- 컴파일된 모델은 캐싱되어 재사용
- 컴파일 캐시 위치: `/tmp/neuron_cache/`

### 2. Tiling 계산

다양한 행렬 크기에 대해 필요한 tile 개수 자동 계산

### 3. Latency 측정

- Warmup iterations으로 안정적인 측정
- 여러 iterations의 평균 latency 계산
- XLA synchronization을 통한 정확한 측정

### 4. 다양한 프로파일링 모드

- **`llm`**: LLM inference를 위한 typical dimensions (prefill/decode)
- **`tiling`**: Tiling boundary 중심 테스트
- **`sweep`**: 전체 dimension 범위 스윕

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

## 중요 사항

### 1. 컴파일 시간

- 최초 실행 시 각 행렬 크기별로 컴파일이 수행됩니다
- 컴파일 시간은 행렬 크기에 따라 수십 초에서 수 분 소요될 수 있습니다
- 컴파일된 모델은 캐싱되므로 이후 실행은 빠릅니다

### 2. XLA Device

- Neuron은 XLA 백엔드를 사용합니다
- `xm.mark_step()` 및 `xm.wait_device_ops()`를 통해 정확한 타이밍 측정

### 3. 메모리 제약

- Neuron 디바이스의 메모리 제약을 고려하여 행렬 크기를 설정하세요
- 너무 큰 행렬은 컴파일 실패하거나 OOM 에러가 발생할 수 있습니다

## 문제 해결

### Neuron 디바이스를 찾을 수 없는 경우

```bash
# Neuron 런타임 확인
ls /opt/aws/neuron/lib/libnrt.so.1

# Neuron 디바이스 확인 (드라이버 설치된 경우)
neuron-ls
```

### 컴파일 에러

- 행렬 크기를 줄여서 시도해보세요
- `/tmp/neuron_cache/` 디렉토리를 삭제하고 재시도하세요

```bash
rm -rf /tmp/neuron_cache/
```

### CPU fallback

Neuron 디바이스가 없으면 자동으로 CPU로 fallback됩니다:

```
Warning: Neuron device not available: ... Falling back to CPU.
Using CPU device
```

## 참고 자료

- [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [PyTorch Neuron Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/)
- [Inferentia2/Trainium Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/)

## vLLM Bucketing 참고

vLLM은 다음과 같이 bucketing을 수행합니다:

- Sequence length bucketing: [128, 256, 512, 1024, 2048]
- Batch size bucketing: power of 2

이를 참고하여 profiling 결과를 분석하고 최적의 bucket을 설정할 수 있습니다.
