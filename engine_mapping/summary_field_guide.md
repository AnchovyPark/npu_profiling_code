# Neuron Profile Summary 핵심 필드 가이드

> 예측 모델에 활용할 수 있는 필드만 정리.
> matmul(compute-bound)과 layernorm(memory-bound) 예시값 포함.

---

## 1. 시간 — 예측 타겟 & 분해

### `total_time`
- **matmul:** `0.01163초 (11.6ms)`
- **layernorm:** `0.00264초 (2.6ms)`
- 디바이스에서의 총 실행 시간. **이게 우리가 예측하려는 값.**

### `total_active_time`
- **matmul:** `0.00979초 (84%)`
- **layernorm:** `0.00078초 (30%)`
- 엔진/DMA 중 하나라도 일하고 있던 시간. total_time과의 차이 = 순수 대기 시간. layernorm은 70%가 대기.

### `tensor_engine_active_time`
- **matmul:** `0.00961초 (82.7%)`
- **layernorm:** `0.00000055초 (0.02%)`
- TensorEngine이 연산 중인 시간. matmul류는 이게 지배적.

### `vector_engine_active_time`
- **matmul:** `0.00000045초 (0.004%)`
- **layernorm:** `0.000317초 (12%)`
- VectorEngine이 연산 중인 시간. norm/activation류는 이게 주력.

### `dma_active_time`
- **matmul:** `0.00202초 (17.4%)`
- **layernorm:** `0.000778초 (29.5%)`
- DMA(메모리 전송)가 활성인 시간. memory-bound 연산일수록 비중 높음.

---

## 2. 연산량 — compute-bound 예측 재료

### `model_flops`
- **matmul:** `137,438,953,472 (137.4 GFLOP)`
- **layernorm:** `0`
- 모델이 요구하는 순수 연산량. VectorEngine 연산(norm, activation)은 0으로 잡힘 — 프로파일러 한계.

### `hardware_flops`
- **matmul:** `309,237,645,312 (309.2 GFLOP)`
- **layernorm:** `0`
- 하드웨어가 실제 수행한 연산량. 패딩/transpose 포함. matmul에서 model 대비 2.25배 → 컴파일러 오버헤드.

### `transpose_flops`
- **matmul:** `34,359,738,368 (34.4 GFLOP)`
- **layernorm:** `0`
- transpose 목적의 MATMUL 연산량. hardware_flops 중 "쓸모없는" 부분.

---

## 3. 메모리 — memory-bound 예측 재료

### `hbm_read_bytes`
- **matmul:** `335,609,856 (320 MiB)`
- **layernorm:** `67,633,152 (64.5 MiB)`
- HBM에서 읽은 총 바이트.

### `hbm_write_bytes`
- **matmul:** `67,108,864 (64 MiB)`
- **layernorm:** `67,108,864 (64 MiB)`
- HBM에 쓴 총 바이트.

→ memory-bound 연산의 시간 ≈ (hbm_read + hbm_write) / 실효 대역폭

### `sbuf_read_bytes`
- **matmul:** `117,440,512 (112 MiB)` — HBM read(320)보다 작음 → **SBUF에서 데이터 재사용**
- **layernorm:** `68,157,952 (65 MiB)` — HBM과 비슷 → **재사용 없음**

### `sbuf_write_bytes`
- **matmul:** `338,231,296 (323 MiB)`
- **layernorm:** `68,124,672 (65 MiB)`

### `psum_read_bytes` / `psum_write_bytes`
- **matmul:** read `2,621,440 (2.5 MiB)` / write `37,748,736 (36 MiB)` — 부분합 누적용
- **layernorm:** 둘 다 `0` — PSUM 미사용

### `spill_save_bytes` / `spill_reload_bytes`
- **matmul:** 둘 다 `0`
- **layernorm:** 둘 다 `0`
- SBUF에 안 들어가서 HBM으로 밀어낸 데이터. **0이 아니면 예측 오차 원인.** 큰 연산에서 발생 가능.

---

## 4. Bound 판별 — compute vs memory

### `mm_arithmetic_intensity`
- **matmul:** `682.6`
- **layernorm:** `0`
- 산술 강도 = (hardware_flops - transpose_flops) / (hbm_read + hbm_write). **peak_flops_bandwidth_ratio와 비교해서 bound 판별.**

### `peak_flops_bandwidth_ratio`
- **값:** `223.8` (하드웨어 고정값)
- 피크 연산력 / 피크 대역폭. **Roofline의 꺾이는 점.**
- arithmetic_intensity > 223.8 → **compute-bound** (matmul: 682 > 223.8 ✅)
- arithmetic_intensity < 223.8 → **memory-bound**

### `mfu_estimated_percent`
- **matmul:** `12.9%`
- **layernorm:** `0%`
- Model FLOPs 기준 TensorEngine 활용률. 피크 대비 실제 얼마나 쓰는지.

### `mbu_estimated_percent`
- **matmul:** `8.5%`
- **layernorm:** `12.5%`
- HBM 대역폭 활용률. 피크 820 GiB/s 대비 실제 사용 비율. 이걸로 **실효 대역폭 역산** 가능.

---

## 5. 보정 요소 — 예측 오차 원인

### `throttle_avg_util_limit_nc0_percent`
- **matmul:** `0.586 (58.6%)` — **성능이 58.6%로 제한되고 있음!**
- **layernorm:** `1.0 (100%)` — 스로틀링 없음
- 1.0이면 제한 없음. 낮을수록 열 등의 이유로 성능 깎임. 예측 모델에서 보정 필요.

### hardware_flops / model_flops 비율
- **matmul:** 2.25배 — 컴파일러가 패딩/transpose로 연산을 2배 이상 뻥튀기
- 이 비율이 연산마다, 크기마다 다르면 예측 오차 원인

---

## 6. 연산 분류 기준

| 지표 | 높으면 | 예시 |
|------|--------|------|
| `tensor_engine_active_time_percent` | TensorEngine 주력 (matmul류) | matmul: 82.7% |
| `vector_engine_active_time_percent` | VectorEngine 주력 (norm/activation류) | layernorm: 12% |
| `matmul_instruction_count` | matmul 연산 포함 | matmul: 24,576 / layernorm: 0 |

---

## 예측 모델에서의 활용 흐름

```
[입력: 연산 종류, 텐서 크기]
        ↓
[1단계: 연산 분류] — engine active % 로 어떤 엔진 주력인지
        ↓
[2단계: bound 판별] — arithmetic_intensity vs peak_ratio(223.8)
        ↓
[3단계: 시간 예측]
  - compute-bound → model_flops / 실효 연산력
  - memory-bound  → (hbm_read + hbm_write) / 실효 대역폭
        ↓
[4단계: 보정] — spill, throttling, 패딩 오버헤드
        ↓
[출력: 예측 시간]
```
