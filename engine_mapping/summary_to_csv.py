#!/usr/bin/env python3
"""
summary_to_csv.py - 여러 JSON 프로파일의 summary를 하나의 CSV로 정리
"""

import json
import csv
import os
import glob

# ============================================================
# 설정: 여기서 입력/출력 지정
# ============================================================
INPUT_DIR = "/Users/parkjuhyun/Desktop/project_chip/npu_profiling_code/single_kernel/results"
OUTPUT_DIR = "/Users/parkjuhyun/Desktop/project_chip/npu_profiling_code/engine_mapping/results"
OUTPUT_FILENAME = "single_kernel_summary.csv"
# ============================================================


# 뽑을 컬럼 정의 (summary[0]의 key, 한글 설명)
COLUMNS = [
    # === 시간 ===
    ("total_time", "전체 실행 시간 (초)"),
    ("total_active_time", "활성 시간 (초)"),
    ("tensor_engine_active_time", "TensorEngine 활성 시간 (초)"),
    ("vector_engine_active_time", "VectorEngine 활성 시간 (초)"),
    ("scalar_engine_active_time", "ScalarEngine 활성 시간 (초)"),
    ("gpsimd_engine_active_time", "GPSIMD 활성 시간 (초)"),
    ("sync_engine_active_time", "SyncEngine 활성 시간 (초)"),
    ("dma_active_time", "DMA 활성 시간 (초)"),
    
    # === 시간 비율 ===
    ("total_active_time_percent", "활성 시간 비율"),
    ("tensor_engine_active_time_percent", "TensorEngine 비율"),
    ("vector_engine_active_time_percent", "VectorEngine 비율"),
    ("scalar_engine_active_time_percent", "ScalarEngine 비율"),
    ("gpsimd_engine_active_time_percent", "GPSIMD 비율"),
    ("dma_active_time_percent", "DMA 비율"),
    
    # === Instruction Count ===
    ("tensor_engine_instruction_count", "TensorEngine 명령어 수"),
    ("vector_engine_instruction_count", "VectorEngine 명령어 수"),
    ("scalar_engine_instruction_count", "ScalarEngine 명령어 수"),
    ("gpsimd_engine_instruction_count", "GPSIMD 명령어 수"),
    ("sync_engine_instruction_count", "SyncEngine 명령어 수"),
    ("matmul_instruction_count", "matmul 명령어 수"),
    
    # === Instruction Time ===
    ("tensor_engine_instruction_time", "TensorEngine 명령어 시간 (초)"),
    ("vector_engine_instruction_time", "VectorEngine 명령어 시간 (초)"),
    ("scalar_engine_instruction_time", "ScalarEngine 명령어 시간 (초)"),
    ("gpsimd_engine_instruction_time", "GPSIMD 명령어 시간 (초)"),
    
    # === 메모리 ===
    ("hbm_read_bytes", "HBM 읽기 (bytes)"),
    ("hbm_write_bytes", "HBM 쓰기 (bytes)"),
    ("sbuf_read_bytes", "SBUF 읽기 (bytes)"),
    ("sbuf_write_bytes", "SBUF 쓰기 (bytes)"),
    ("psum_read_bytes", "PSUM 읽기 (bytes)"),
    ("psum_write_bytes", "PSUM 쓰기 (bytes)"),
    ("weight_size_bytes", "Weight 크기 (bytes)"),
    ("spill_reload_bytes", "Spill reload (bytes)"),
    ("spill_save_bytes", "Spill save (bytes)"),
    
    # === 연산 효율 ===
    ("model_flops", "Model FLOPs"),
    ("hardware_flops", "Hardware FLOPs"),
    ("mfu_estimated_percent", "MFU (Model FLOPs Utilization)"),
    ("hfu_estimated_percent", "HFU (Hardware FLOPs Utilization)"),
    ("mbu_estimated_percent", "MBU (Memory Bandwidth Utilization)"),
    ("mm_arithmetic_intensity", "산술 강도 (FLOPs/Byte)"),
    ("peak_flops_bandwidth_ratio", "피크 산술 강도"),
    
    # === DMA ===
    ("dma_transfer_count", "DMA 전송 횟수"),
    ("dma_transfer_total_bytes", "DMA 전송량 (bytes)"),
    ("dma_transfer_time", "DMA 전송 시간 (초)"),
    
    # === 스로틀링 ===
    ("throttle_avg_util_limit_nc0_percent", "스로틀링 제한 비율"),
    
    # === 기타 ===
    ("neuroncore_cycle_count", "NeuronCore 사이클 수"),
    ("instance_type", "인스턴스 타입"),
]


def extract_op_name(filename):
    """파일명에서 연산 이름 추출."""
    name = os.path.basename(filename)
    name = name.replace('profile_', '').replace('.json', '')
    return name


def main():
    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    
    if not json_files:
        print(f"❌ JSON 파일 없음: {INPUT_DIR}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    col_keys = [c[0] for c in COLUMNS]
    header = ["operation"] + col_keys
    
    rows = []
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            s = data['summary'][0]
            op_name = extract_op_name(jf)
            row = [op_name]
            for key in col_keys:
                row.append(s.get(key, ''))
            rows.append(row)
        except Exception as e:
            print(f"⚠️ {jf} 읽기 실패: {e}")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        desc_header = ["operation"] + [c[1] for c in COLUMNS]
        writer.writerow(desc_header)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    
    print(f"✅ CSV 저장: {output_path}")
    print(f"   파일 수: {len(rows)}개")
    print(f"   컬럼 수: {len(col_keys)}개")
    print(f"\n📋 포함된 연산:")
    for row in rows:
        print(f"   {row[0]}")


if __name__ == "__main__":
    main()
