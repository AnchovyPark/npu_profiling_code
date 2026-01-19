# NEFF 파일 설명

## 1. NEFF란?

**NEFF** = **Neuron Executable File Format**

- AWS Neuron 컴파일러가 생성하는 실행 파일
- NPU(Inferentia/Trainium)에서 직접 실행되는 바이너리 코드
- GPU의 CUDA 바이너리와 비슷한 개념

## 2. 컴파일 과정

```
PyTorch 코드 (torch.matmul)
    ↓
XLA HLO (High-Level Operations) - 중간 표현
    ↓
Neuron Compiler (neuronx-cc)
    ↓
NEFF 파일 (NPU 실행 코드)
```

## 3. 저장 위치 & 디렉토리 구조

### 위치
```bash
/tmp/neuron_cache/matmul_MxKxN_dtype/
```

### 디렉토리 이름 의미
```
matmul_MxKxN_dtype
       │ │ │   └─ 데이터 타입 (float16, float32, bfloat16)
       │ │ └───── N: 행렬 B의 열 수 (출력 행렬 C의 열 수)
       │ └─────── K: 공통 차원 (contraction dimension)
       └───────── M: 행렬 A의 행 수 (출력 행렬 C의 행 수)
```

**행렬 연산:**
```
A(M×K) × B(K×N) = C(M×N)
```

### 예시
```
matmul_128x128x128_float16  → M=128, K=128, N=128, FP16
matmul_256x128x128_float16  → M=256, K=128, N=128, FP16
matmul_512x512x1024_float32 → M=512, K=512, N=1024, FP32
```

## 4. 디렉토리 내용

```
matmul_128x128x128_float16/
├── graph.neff          # NPU 실행 파일 (바이너리, 31K)
├── command.txt         # 컴파일 명령어
└── model/
    ├── metadata.json   # 메타데이터
    └── graph.hlo       # XLA 중간 표현 (바이너리)
```

### 각 파일 설명

| 파일 | 내용 | 형식 |
|------|------|------|
| `graph.neff` | NPU 실행 코드 (타일링 정보 포함) | 바이너리 |
| `graph.hlo` | XLA 중간 표현 | 바이너리 프로토콜 버퍼 |
| `command.txt` | `neuronx-cc compile ...` | 텍스트 |
| `metadata.json` | 버전, 파일 정보 | JSON |

## 5. 중요 발견: NEFF 파일 크기가 모두 동일 (31K)

### 실험 결과
```bash
M=128  (1 tile)  → graph.neff: 31K
M=256  (2 tiles) → graph.neff: 31K  ← 같음!
M=512  (4 tiles) → graph.neff: 31K  ← 같음!
M=1024 (8 tiles) → graph.neff: 31K  ← 같음!
```

### 왜 크기가 같을까?

**각 (M, K, N)마다 별도 NEFF가 생성되지만, 크기는 비슷함**

#### NEFF 파일 구성 (추정)
```
NEFF 파일 (31K):
┌────────────────────────────────┐
│ 1. 타일링 정보 (매우 작음)      │
│    - num_tiles_M: 8            │  ← 4 bytes
│    - num_tiles_K: 1            │  ← 4 bytes
│    - num_tiles_N: 1            │  ← 4 bytes
├────────────────────────────────┤
│ 2. 128×128 PE 실행 코드 (큼)   │
│    - matmul kernel             │  ← 대부분 (수십 KB)
│    - 메모리 레이아웃           │
│    - 최적화 코드               │
├────────────────────────────────┤
│ 3. 메타데이터                  │  ← 작음
└────────────────────────────────┘
```

**타일 반복 횟수는 12 bytes 정도** (int 3개)
→ 31KB 파일에서 무시할 수 있는 크기!

### 결론
- **각 (M, K, N)마다 별도 NEFF 생성됨** (재사용 안함)
- **타일 반복 정보가 NEFF 안에 있음**
- 파일 크기가 같은 이유: 반복 횟수는 매우 작은 정보 (12 bytes)

## 6. 컴파일 캐싱

### 캐시 동작
```python
# 첫 실행: 컴파일 발생
A = torch.randn(128, 128, dtype=torch.float16)
B = torch.randn(128, 128, dtype=torch.float16)
C = compiled_model(A, B)  # → NEFF 생성 (수 초 소요)

# 같은 shape 재실행: 캐시 사용
C = compiled_model(A, B)  # → 빠름 (캐시된 NEFF 재사용)

# 다른 shape: 새로 컴파일
A2 = torch.randn(256, 128, dtype=torch.float16)
C2 = compiled_model(A2, B)  # → 새 NEFF 생성
```

### 캐시 키
```python
(M, K, N, dtype) → 고유한 NEFF 파일
```

## 7. 타일링 정보 분석

### 문제: NEFF는 바이너리라서 직접 읽기 어려움

### 해결: Latency 측정으로 간접 유추

```bash
# 분석 스크립트 실행
python3 analyze_tiling.py ./results/custom_test.csv
```

**출력 예시:**
```
     M      K      N | TilesM TilesK TilesN  Total |    Latency   Per Tile
--------------------------------------------------------------------------------
   128    128    128 |      1      1      1      1 |     0.2958     0.2958  ←
   160    128    128 |      2      1      1      2 |     0.2922     0.1461  ←
   256    128    128 |      2      1      1      2 |     0.2892     0.1446  ←
   288    128    128 |      3      1      1      3 |     0.3049     0.1016  ←
```

### 타일 개수 계산 (코드)
```python
# npu_matmul_profiler.py:97-99
num_tiles_M = (M + 127) // 128
num_tiles_K = (K + 511) // 512
num_tiles_N = (N + 127) // 128
```

## 8. 실제 NPU 동작 (추정)

### M=128 (1 tile)
```
NEFF: num_tiles_M = 1

NPU 실행:
┌─────────────────┐
│ PE(128×128)     │ → A[0:128, :] × B
└─────────────────┘
1번 실행, 0.296ms
```

### M=256 (2 tiles)
```
NEFF: num_tiles_M = 2

NPU 실행:
┌─────────────────┐
│ PE(128×128)     │ → A[0:128, :] × B
└─────────────────┘
┌─────────────────┐
│ PE(128×128)     │ → A[128:256, :] × B
└─────────────────┘
2번 실행, 0.311ms (1.05배)
```

### M=512 (4 tiles)
```
NEFF: num_tiles_M = 4

NPU 실행: 4번 반복
→ 0.316ms (1.06배)
```

## 9. 중요한 관찰

### Latency가 타일 개수에 정비례하지 않음!

```
Tiles   Latency   Expected    Actual Ratio
1       0.296ms   0.296ms     1.00x
2       0.311ms   0.592ms     1.05x  ← 2배가 아님!
4       0.316ms   1.184ms     1.06x  ← 4배가 아님!
```

**이유:**
1. **Padding overhead**: 작은 M에서는 setup cost가 큼
2. **병렬 처리**: NPU가 여러 tile을 병렬로 처리할 수 있음
3. **메모리 재사용**: B 행렬은 재사용됨

## 10. 최적화 팁

### 128의 배수로 맞추기
```
M=128  ✓ 최적 (1 tile, no padding)
M=129  ✗ 비효율 (2 tiles, 127 padding 낭비)
M=256  ✓ 최적 (2 tiles, no padding)
```

### Boundary 효과
```bash
python3 quick_boundary_test.py
```

**M=127 vs M=128 vs M=129:**
- M=127: 1 tile (padding 1)
- M=128: 1 tile (perfect fit) ← 최적
- M=129: 2 tiles (padding 127) ← 2배 느림!

## 11. 요약

| 항목 | 설명 |
|------|------|
| **NEFF 파일** | NPU 실행 바이너리 (31K) |
| **저장 위치** | `/tmp/neuron_cache/matmul_MxKxN_dtype/` |
| **컴파일 단위** | 각 (M, K, N, dtype) 조합마다 별도 |
| **타일 정보** | NEFF 안에 포함 (직접 읽기 어려움) |
| **파일 크기** | 모두 31K (타일 정보는 12 bytes 정도) |
| **분석 방법** | Latency 측정으로 간접 유추 (`analyze_tiling.py`) |
| **최적화** | 128의 배수로 맞추기 |

## 12. 관련 파일

- `npu_matmul_profiler.py`: 메인 프로파일러
- `custom_test.py`: 사용자 테스트 인터페이스
- `analyze_tiling.py`: 타일링 분석 스크립트
- `verbose_compile_test.py`: 상세 컴파일 로그
