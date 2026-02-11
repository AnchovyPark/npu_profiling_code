#!/bin/bash
# Single Kernel Neuron Profiling Script
#
# 각 커널별 Python 스크립트 실행 → NEFF 생성 → neuron-profile capture → JSON 추출
#
# Usage:
#   ./run_neuron_profile.sh          # 전체 커널 프로파일링
#   ./run_neuron_profile.sh matmul   # 특정 커널만 프로파일링
#   ./run_neuron_profile.sh "matmul add silu"  # 여러 커널 지정

set -e

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/neuron_profile_results"

# 전체 커널 목록
ALL_KERNELS="matmul add mul silu gelu rmsnorm layernorm softmax rope"

# 인자가 있으면 해당 커널만, 없으면 전체
if [ -n "$1" ]; then
    KERNELS="$1"
else
    KERNELS="$ALL_KERNELS"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Single Kernel Neuron Profiling${NC}"
echo -e "${GREEN}  Kernels: ${KERNELS}${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

mkdir -p "${OUTPUT_DIR}"

TOTAL_JSON=0

for KERNEL in $KERNELS; do
    SCRIPT="${SCRIPT_DIR}/${KERNEL}.py"

    if [ ! -f "$SCRIPT" ]; then
        echo -e "${RED}[SKIP] ${KERNEL}.py not found${NC}"
        continue
    fi

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Kernel: ${KERNEL}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Step 0: 이전 NEFF 정리
    echo -e "${YELLOW}[0] 기존 NEFF 정리 (neuron_single_${KERNEL}_*)${NC}"
    rm -rf /tmp/neuron_single_${KERNEL}_*

    # Step 1: Python 스크립트 실행 → NEFF 생성
    echo -e "${YELLOW}[1] ${KERNEL}.py 실행 중...${NC}"
    python3 "${SCRIPT}"
    echo -e "${GREEN}[1] 완료${NC}"
    echo ""

    # Step 2: NEFF 디렉토리 찾기
    NEFF_DIRS=$(find /tmp -maxdepth 1 -type d -name "neuron_single_${KERNEL}_*" 2>/dev/null | sort)

    if [ -z "$NEFF_DIRS" ]; then
        echo -e "${RED}  NEFF 디렉토리 없음. 건너뜁니다.${NC}"
        echo ""
        continue
    fi

    NEFF_COUNT=$(echo "$NEFF_DIRS" | wc -l)
    echo -e "${GREEN}[2] ${NEFF_COUNT}개 NEFF 디렉토리 발견${NC}"

    # Step 3 & 4: 각 NEFF에 대해 프로파일링
    for COMPILER_DIR in $NEFF_DIRS; do
        DIR_NAME=$(basename "$COMPILER_DIR")
        SHAPE=$(echo "$DIR_NAME" | sed "s/neuron_single_${KERNEL}_//")

        echo -e "  ${YELLOW}[3] ${KERNEL}_${SHAPE}: neuron-profile capture...${NC}"

        NEFF_FILE=$(find "$COMPILER_DIR" -name "*.neff" -type f | head -n 1)

        if [ -z "$NEFF_FILE" ]; then
            echo -e "  ${RED}NEFF 파일 없음. 건너뜁니다.${NC}"
            continue
        fi

        NTFF_FILE="${OUTPUT_DIR}/profile_${KERNEL}_${SHAPE}.ntff"
        JSON_FILE="${OUTPUT_DIR}/profile_${KERNEL}_${SHAPE}.json"

        # Capture
        neuron-profile capture \
            -n "${NEFF_FILE}" \
            -s "${NTFF_FILE}" 2>&1 | grep -v "level=info" || true

        if [ ! -f "${NTFF_FILE}" ]; then
            echo -e "  ${RED}NTFF 생성 실패. 건너뜁니다.${NC}"
            continue
        fi

        # View → JSON
        echo -e "  ${YELLOW}[4] ${KERNEL}_${SHAPE}: neuron-profile view...${NC}"
        neuron-profile view \
            -n "${NEFF_FILE}" \
            -s "${NTFF_FILE}" \
            --output-format json \
            --output-file "${JSON_FILE}" 2>&1 | grep -v "level=info" || true

        if [ -f "${JSON_FILE}" ]; then
            echo -e "  ${GREEN}OK: ${JSON_FILE} ($(du -h "${JSON_FILE}" | cut -f1))${NC}"
            TOTAL_JSON=$((TOTAL_JSON + 1))
        else
            echo -e "  ${RED}JSON 생성 실패${NC}"
        fi
    done

    # NEFF 정리 (디스크 절약)
    rm -rf /tmp/neuron_single_${KERNEL}_*
    echo ""
done

# 결과 요약
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}프로파일링 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "출력 디렉토리: ${OUTPUT_DIR}"
echo ""
echo -e "${YELLOW}생성된 파일 목록:${NC}"
ls -lh "${OUTPUT_DIR}/" 2>/dev/null || echo "파일이 없습니다"
echo ""
echo -e "${GREEN}총 ${TOTAL_JSON}개의 JSON 프로파일 생성됨${NC}"
