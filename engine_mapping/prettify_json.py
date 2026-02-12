#!/usr/bin/env python3
"""
prettify_json.py - JSON 파일을 보기 좋게 줄바꿈해서 저장
"""

import json
import os
import glob

# ============================================================
# 설정: 여기서 입력 파일/폴더 지정
# ============================================================
INPUT_DIR = "/Users/parkjuhyun/Desktop/project_chip/npu_profiling_code/single_kernel/results"
OUTPUT_DIR = "/Users/parkjuhyun/Desktop/project_chip/npu_profiling_code/engine_mapping/results"

# 펼칠 섹션 (이것만 보기 좋게 표시, 나머지는 전부 생략)
SHOW_KEYS = [
    "summary",
    "neff_node",
    "nc_mem_usage",
]
# ============================================================


def custom_dump(data, f, indent=2):
    """생략 대상 섹션은 한 줄로, 나머지는 보기 좋게 펼쳐서 저장."""
    f.write("{\n")
    keys = list(data.keys())
    for i, k in enumerate(keys):
        v = data[k]
        comma = "," if i < len(keys) - 1 else ""
        
        if k not in SHOW_KEYS:
            count = len(v) if isinstance(v, list) else "?"
            f.write(f'  "{k}": ["... {count}개 항목 생략 ..."]{comma}\n')
        else:
            dumped = json.dumps(v, indent=indent, ensure_ascii=False)
            # indent 맞추기 (첫 줄 제외 2칸 들여쓰기)
            lines = dumped.split("\n")
            if len(lines) == 1:
                f.write(f'  "{k}": {dumped}{comma}\n')
            else:
                indented = lines[0] + "\n" + "\n".join("  " + line for line in lines[1:])
                f.write(f'  "{k}": {indented}{comma}\n')
    f.write("}\n")


def main():
    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    
    if not json_files:
        print(f"❌ JSON 파일 없음: {INPUT_DIR}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for jf in json_files:
        fname = os.path.basename(jf)
        base, ext = os.path.splitext(fname)
        output_path = os.path.join(OUTPUT_DIR, f"{base}_pretty{ext}")
        
        with open(jf, 'r') as f:
            data = json.load(f)
        
        with open(output_path, 'w') as f:
            custom_dump(data, f)
        
        input_size = os.path.getsize(jf) / 1024 / 1024
        output_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"  ✅ {fname} ({input_size:.1f}MB) → {os.path.basename(output_path)} ({output_size:.1f}MB)")
    
    print(f"\n📁 출력 폴더: {OUTPUT_DIR}")
    print(f"   총 {len(json_files)}개 변환 완료")


if __name__ == "__main__":
    main()
