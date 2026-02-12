#!/usr/bin/env python3
"""
Cross-Engine Dependent vs Independent 비교 분석
각 pair별 CSV 파일 생성 + 전체 요약 CSV
"""

import json
import csv
import os

# ============================================================
# 설정
# ============================================================
BASE_DIR = "/Users/parkjuhyun/Desktop/neuron_profile_results"
DEP_DIR = os.path.join(BASE_DIR, "dependent")
IND_DIR = os.path.join(BASE_DIR, "independent")
OUTPUT_DIR = os.path.join(BASE_DIR, "comparison")

PAIRS = [
    "matmul_add", "matmul_silu", "matmul_layernorm", "matmul_rmsnorm", "matmul_softmax",
    "layernorm_matmul", "rmsnorm_matmul", "softmax_matmul",
    "add_layernorm", "add_rmsnorm", "silu_mul"
]

SHAPE = "4096x4096"

# 비교할 summary 필드
FIELDS = [
    # 시간 (원본 sec → μs로 변환)
    "total_time",
    "total_active_time",
    # 엔진 active time (5개)
    "tensor_engine_active_time",
    "vector_engine_active_time",
    "scalar_engine_active_time",
    "gpsimd_engine_active_time",
    "dma_active_time",
    # FLOPs
    "hardware_flops",
    # 메모리
    "hbm_read_bytes",
    "hbm_write_bytes",
    "sbuf_read_bytes",
    "sbuf_write_bytes",
    "psum_read_bytes",
    "psum_write_bytes",
    "psum_read_sbuf_write_bytes",
]

# sec → μs 변환 대상 필드
TIME_FIELDS = {
    "total_time", "total_active_time",
    "tensor_engine_active_time", "vector_engine_active_time",
    "scalar_engine_active_time", "gpsimd_engine_active_time",
    "dma_active_time",
}

# CSV column 이름에 단위 표시
def col_name(field):
    if field in TIME_FIELDS:
        return field + "_us"
    return field


def load_summary(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data["summary"][0]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []

    for pair in PAIRS:
        dep_file = os.path.join(DEP_DIR, f"profile_{pair}_{SHAPE}.json")
        ind_file = os.path.join(IND_DIR, f"profile_{pair}_{SHAPE}.json")

        if not os.path.exists(dep_file) or not os.path.exists(ind_file):
            print(f"[SKIP] {pair}: 파일 없음")
            continue

        print(f"[{pair}] 로딩 중...", end=" ", flush=True)
        dep_s = load_summary(dep_file)
        ind_s = load_summary(ind_file)
        print("OK")

        # pair별 CSV — type이 row, field가 column
        csv_path = os.path.join(OUTPUT_DIR, f"{pair}.csv")
        col_names = [col_name(f) for f in FIELDS]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["type"] + col_names)
            writer.writeheader()
            
            dep_row = {"type": "dependent"}
            ind_row = {"type": "independent"}
            diff_row = {"type": "diff"}
            pct_row = {"type": "diff_%"}
            
            for field in FIELDS:
                cn = col_name(field)
                dep_v = dep_s.get(field, 0)
                ind_v = ind_s.get(field, 0)
                
                # 시간 필드는 sec → μs 변환
                if field in TIME_FIELDS:
                    dep_v = round(dep_v * 1_000_000, 2)
                    ind_v = round(ind_v * 1_000_000, 2)
                
                dep_row[cn] = dep_v
                ind_row[cn] = ind_v
                diff_row[cn] = round(dep_v - ind_v, 2)
                pct_row[cn] = f"{((dep_v - ind_v) / ind_v * 100):.2f}" if ind_v != 0 else "0.00"
            
            writer.writerow(dep_row)
            writer.writerow(ind_row)
            writer.writerow(diff_row)
            writer.writerow(pct_row)

        # 전체 요약용 행 추가
        dep_sum = {"pair": pair, "type": "dependent"}
        ind_sum = {"pair": pair, "type": "independent"}
        for field in FIELDS:
            cn = col_name(field)
            dep_v = dep_s.get(field, 0)
            ind_v = ind_s.get(field, 0)
            if field in TIME_FIELDS:
                dep_v = round(dep_v * 1_000_000, 2)
                ind_v = round(ind_v * 1_000_000, 2)
            dep_sum[cn] = dep_v
            ind_sum[cn] = ind_v
        all_rows.append(dep_sum)
        all_rows.append(ind_sum)

    # 전체 요약 CSV
    col_names = [col_name(f) for f in FIELDS]
    summary_path = os.path.join(OUTPUT_DIR, "all_pairs_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "type"] + col_names)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n출력 디렉토리: {OUTPUT_DIR}")
    print(f"pair별 CSV: {len(PAIRS)}개")
    print(f"전체 요약: {summary_path}")


if __name__ == "__main__":
    main()
