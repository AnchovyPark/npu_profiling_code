#!/usr/bin/env python3
"""
analyze_engine_mapping.py - Neuron Profile JSON 분석

단일 커널 프로파일링 결과에서 각 연산이 어떤 엔진(들)을 사용하는지 파악.
"""

import json
import os
import sys
import csv
from datetime import datetime

# ============================================================
# 설정: 여기서 입력/출력 지정
# ============================================================
INPUT_DIR = "/Users/parkjuhyun/Desktop/project_chip/npu_profiling_code/single_kernel/results"
OUTPUT_DIR = "/Users/parkjuhyun/Desktop/project_chip/npu_profiling_code/engine_mapping/results"
# ============================================================


def load_profile(json_path):
    """JSON 프로파일 로드."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_engine_summary(data):
    """summary에서 엔진별 정보 추출."""
    s = data['summary'][0]
    
    return {
        # Active time (seconds)
        'tensor_engine_time_us':  s.get('tensor_engine_active_time', 0) * 1e6,
        'vector_engine_time_us':  s.get('vector_engine_active_time', 0) * 1e6,
        'scalar_engine_time_us':  s.get('scalar_engine_active_time', 0) * 1e6,
        'gpsimd_engine_time_us':  s.get('gpsimd_engine_active_time', 0) * 1e6,
        'sync_engine_time_us':    s.get('sync_engine_active_time', 0) * 1e6,
        'dma_time_us':            s.get('dma_active_time', 0) * 1e6,
        'total_time_us':          s.get('total_time', 0) * 1e6,
        'total_active_time_us':   s.get('total_active_time', 0) * 1e6,
        
        # Active time percent
        'tensor_engine_pct':  s.get('tensor_engine_active_time_percent', 0) * 100,
        'vector_engine_pct':  s.get('vector_engine_active_time_percent', 0) * 100,
        'scalar_engine_pct':  s.get('scalar_engine_active_time_percent', 0) * 100,
        'gpsimd_engine_pct':  s.get('gpsimd_engine_active_time_percent', 0) * 100,
        'dma_pct':            s.get('dma_active_time_percent', 0) * 100,
        
        # Instruction counts
        'tensor_instr_count': s.get('tensor_engine_instruction_count', 0),
        'vector_instr_count': s.get('vector_engine_instruction_count', 0),
        'scalar_instr_count': s.get('scalar_engine_instruction_count', 0),
        'gpsimd_instr_count': s.get('gpsimd_engine_instruction_count', 0),
        'sync_instr_count':   s.get('sync_engine_instruction_count', 0),
        
        # Instruction time (seconds → us)
        'tensor_instr_time_us': s.get('tensor_engine_instruction_time', 0) * 1e6,
        'vector_instr_time_us': s.get('vector_engine_instruction_time', 0) * 1e6,
        'scalar_instr_time_us': s.get('scalar_engine_instruction_time', 0) * 1e6,
        'gpsimd_instr_time_us': s.get('gpsimd_engine_instruction_time', 0) * 1e6,
        
        # Memory
        'sbuf_read_bytes':  s.get('sbuf_read_bytes', 0),
        'sbuf_write_bytes': s.get('sbuf_write_bytes', 0),
        'hbm_read_bytes':   s.get('hbm_read_bytes', 0),
        'hbm_write_bytes':  s.get('hbm_write_bytes', 0),
        
        # Model info
        'model_flops':      s.get('model_flops', 0),
    }


def classify_engines(info, threshold_pct=1.0):
    """어떤 엔진을 '실질적으로' 사용하는지 분류.
    
    Args:
        info: extract_engine_summary 결과
        threshold_pct: 이 비율(%) 이상이면 '사용'으로 판단
    """
    engines = []
    
    if info['tensor_engine_pct'] >= threshold_pct:
        engines.append('TensorEngine')
    if info['vector_engine_pct'] >= threshold_pct:
        engines.append('VectorEngine')
    if info['scalar_engine_pct'] >= threshold_pct:
        engines.append('ScalarEngine')
    if info['gpsimd_engine_pct'] >= threshold_pct:
        engines.append('GPSIMD')
    
    return engines


def extract_instruction_types(data):
    """instruction 리스트에서 고유한 instruction type 추출."""
    instructions = data.get('instruction', [])
    
    type_counts = {}
    for inst in instructions:
        itype = inst.get('instruction_type', 'unknown')
        opcode = inst.get('opcode', 'unknown')
        key = f"{itype}"
        type_counts[key] = type_counts.get(key, 0) + 1
    
    return type_counts


def print_report(profile_dir):
    """프로파일 분석 리포트 출력."""
    
    json_files = sorted([f for f in os.listdir(profile_dir) if f.endswith('.json')])
    
    if not json_files:
        print(f"No JSON files found in {profile_dir}")
        return
    
    results = {}
    
    for fname in json_files:
        # 연산 이름 추출: profile_add_4096x4096.json → add
        op_name = fname.replace('profile_', '').replace('_4096x4096.json', '').replace('.json', '')
        
        data = load_profile(os.path.join(profile_dir, fname))
        info = extract_engine_summary(data)
        engines_used = classify_engines(info)
        instr_types = extract_instruction_types(data)
        
        results[op_name] = {
            'info': info,
            'engines': engines_used,
            'instr_types': instr_types,
        }
    
    # ============================================================
    # Report 1: 엔진 사용 시간 (절대값)
    # ============================================================
    print("=" * 110)
    print("📊 엔진별 Active Time (μs)")
    print("=" * 110)
    print(f"{'Operation':<12} {'TensorEng':>10} {'VectorEng':>10} {'ScalarEng':>10} {'GPSIMD':>10} {'DMA':>10} {'Total':>10}")
    print("-" * 110)
    
    for op, r in sorted(results.items()):
        i = r['info']
        print(f"{op:<12} {i['tensor_engine_time_us']:>10.1f} {i['vector_engine_time_us']:>10.1f} "
              f"{i['scalar_engine_time_us']:>10.1f} {i['gpsimd_engine_time_us']:>10.1f} "
              f"{i['dma_time_us']:>10.1f} {i['total_time_us']:>10.1f}")
    
    # ============================================================
    # Report 2: 엔진 사용 비율 (%)
    # ============================================================
    print()
    print("=" * 90)
    print("📊 엔진별 Active Time (%)")
    print("=" * 90)
    print(f"{'Operation':<12} {'TensorEng':>10} {'VectorEng':>10} {'ScalarEng':>10} {'GPSIMD':>10} {'DMA':>10}")
    print("-" * 90)
    
    for op, r in sorted(results.items()):
        i = r['info']
        print(f"{op:<12} {i['tensor_engine_pct']:>9.1f}% {i['vector_engine_pct']:>9.1f}% "
              f"{i['scalar_engine_pct']:>9.1f}% {i['gpsimd_engine_pct']:>9.1f}% "
              f"{i['dma_pct']:>9.1f}%")
    
    # ============================================================
    # Report 3: Instruction Count
    # ============================================================
    print()
    print("=" * 90)
    print("📊 엔진별 Instruction Count")
    print("=" * 90)
    print(f"{'Operation':<12} {'TensorEng':>10} {'VectorEng':>10} {'ScalarEng':>10} {'GPSIMD':>10} {'Sync':>10}")
    print("-" * 90)
    
    for op, r in sorted(results.items()):
        i = r['info']
        print(f"{op:<12} {i['tensor_instr_count']:>10} {i['vector_instr_count']:>10} "
              f"{i['scalar_instr_count']:>10} {i['gpsimd_instr_count']:>10} "
              f"{i['sync_instr_count']:>10}")
    
    # ============================================================
    # Report 4: 엔진 분류 결과 (핵심!)
    # ============================================================
    print()
    print("=" * 90)
    print("🔬 연산별 엔진 분류 (active time > 1% 기준)")
    print("=" * 90)
    
    for op, r in sorted(results.items()):
        engines = r['engines']
        is_single = len(engines) == 1
        marker = "✅ 단일 엔진" if is_single else "⚠️ 복합 엔진"
        print(f"  {op:<12} → {', '.join(engines):<40} {marker}")
    
    # ============================================================
    # Report 5: Cross-Engine 실험 적합성
    # ============================================================
    print()
    print("=" * 90)
    print("🎯 Cross-Engine 전환 비용 측정에 적합한 연산")
    print("=" * 90)
    
    single_tensor = []
    single_vector = []
    single_scalar = []
    multi_engine = []
    
    for op, r in sorted(results.items()):
        engines = r['engines']
        if len(engines) == 1:
            if 'TensorEngine' in engines:
                single_tensor.append(op)
            elif 'VectorEngine' in engines:
                single_vector.append(op)
            elif 'ScalarEngine' in engines:
                single_scalar.append(op)
        else:
            multi_engine.append((op, engines))
    
    print(f"\n  ✅ TensorEngine 단일: {', '.join(single_tensor) if single_tensor else '없음'}")
    print(f"  ✅ VectorEngine 단일: {', '.join(single_vector) if single_vector else '없음'}")
    print(f"  ✅ ScalarEngine 단일: {', '.join(single_scalar) if single_scalar else '없음'}")
    print(f"\n  ⚠️ 복합 엔진 (cross-engine 측정에 부적합):")
    for op, engines in multi_engine:
        print(f"     {op}: {', '.join(engines)}")
    
    # ============================================================
    # Report 6: Memory Usage
    # ============================================================
    print()
    print("=" * 90)
    print("📊 Memory Access (bytes)")
    print("=" * 90)
    print(f"{'Operation':<12} {'SBUF Read':>14} {'SBUF Write':>14} {'HBM Read':>14} {'HBM Write':>14}")
    print("-" * 90)
    
    for op, r in sorted(results.items()):
        i = r['info']
        print(f"{op:<12} {i['sbuf_read_bytes']:>14,} {i['sbuf_write_bytes']:>14,} "
              f"{i['hbm_read_bytes']:>14,} {i['hbm_write_bytes']:>14,}")
    
    # ============================================================
    # Save CSV
    # ============================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"engine_mapping_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    fieldnames = [
        'operation', 'engines_used', 'is_single_engine',
        'tensor_engine_time_us', 'vector_engine_time_us', 'scalar_engine_time_us',
        'gpsimd_engine_time_us', 'dma_time_us', 'total_time_us',
        'tensor_engine_pct', 'vector_engine_pct', 'scalar_engine_pct', 'gpsimd_engine_pct',
        'tensor_instr_count', 'vector_instr_count', 'scalar_instr_count', 'gpsimd_instr_count',
        'sbuf_read_bytes', 'sbuf_write_bytes', 'hbm_read_bytes', 'hbm_write_bytes',
        'model_flops',
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for op, r in sorted(results.items()):
            i = r['info']
            row = {
                'operation': op,
                'engines_used': '|'.join(r['engines']),
                'is_single_engine': len(r['engines']) == 1,
            }
            for k in fieldnames[3:]:
                row[k] = round(i.get(k, 0), 4)
            writer.writerow(row)
    
    print(f"\n📁 CSV 저장: {csv_path}")
    
    return results


if __name__ == "__main__":
    print(f"Profile directory: {INPUT_DIR}\n")
    print_report(INPUT_DIR)
