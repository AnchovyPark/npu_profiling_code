#!/bin/bash
# Cross-Engine Independent Pair Neuron Profiling Script
#
# 각 cross-engine 조합을 비종속적(independent)으로 실행 → NEFF 생성 → neuron-profile capture → JSON 추출
# dependent 버전과 비교하여 엔진 전환 오버헤드를 측정하기 위함
#
# Usage:
#   ./run_neuron_profile.sh                        # 전체 조합 프로파일링
#   ./run_neuron_profile.sh matmul_add             # 특정 조합만
#   ./run_neuron_profile.sh "matmul_add matmul_silu"  # 여러 조합 지정

set -e

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/neuron_profile_results"

# ============================================================
# 전체 조합 목록 (dependent 버전과 동일)
# ============================================================
ALL_PAIRS="matmul_add matmul_silu matmul_layernorm matmul_rmsnorm matmul_softmax layernorm_matmul rmsnorm_matmul softmax_matmul add_layernorm add_rmsnorm silu_mul"

# 인자가 있으면 해당 조합만, 없으면 전체
if [ -n "$1" ]; then
    PAIRS="$1"
else
    PAIRS="$ALL_PAIRS"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cross-Engine INDEPENDENT Pair Profiling${NC}"
echo -e "${GREEN}  Pairs: ${PAIRS}${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

mkdir -p "${OUTPUT_DIR}"

TOTAL_JSON=0

for PAIR in $PAIRS; do
    SCRIPT="${SCRIPT_DIR}/${PAIR}.py"

    if [ ! -f "$SCRIPT" ]; then
        echo -e "${RED}[SKIP] ${PAIR}.py not found${NC}"
        continue
    fi

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Pair: ${PAIR} (independent)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Step 0: 이전 NEFF 정리
    echo -e "${YELLOW}[0] 기존 NEFF 정리 (neuron_indep_${PAIR}_*)${NC}"
    rm -rf /tmp/neuron_indep_${PAIR}_*

    # Step 1: Python 스크립트 실행 → NEFF 생성
    echo -e "${YELLOW}[1] ${PAIR}.py 실행 중...${NC}"
    python3 "${SCRIPT}"
    echo -e "${GREEN}[1] 완료${NC}"
    echo ""

    # Step 2: NEFF 디렉토리 찾기
    NEFF_DIRS=$(find /tmp -maxdepth 1 -type d -name "neuron_indep_${PAIR}_*" 2>/dev/null | sort)

    if [ -z "$NEFF_DIRS" ]; then
        echo -e "${RED}  NEFF 디렉토리 없음. 건너뜁니다.${NC}"
        echo ""
        continue
    fi

    # Step 3 & 4: 각 NEFF에 대해 프로파일링
    for COMPILER_DIR in $NEFF_DIRS; do
        DIR_NAME=$(basename "$COMPILER_DIR")
        SHAPE=$(echo "$DIR_NAME" | sed "s/neuron_indep_${PAIR}_//")

        echo -e "  ${YELLOW}[3] ${PAIR}_${SHAPE}: neuron-profile capture...${NC}"

        NEFF_FILE=$(find "$COMPILER_DIR" -name "*.neff" -type f | head -n 1)

        if [ -z "$NEFF_FILE" ]; then
            echo -e "  ${RED}NEFF 파일 없음. 건너뜁니다.${NC}"
            continue
        fi

        NTFF_FILE="${OUTPUT_DIR}/profile_${PAIR}_${SHAPE}.ntff"
        JSON_FILE="${OUTPUT_DIR}/profile_${PAIR}_${SHAPE}.json"

        # Capture
        neuron-profile capture \
            -n "${NEFF_FILE}" \
            -s "${NTFF_FILE}" 2>&1 | grep -v "level=info" || true

        if [ ! -f "${NTFF_FILE}" ]; then
            echo -e "  ${RED}NTFF 생성 실패. 건너뜁니다.${NC}"
            continue
        fi

        # View → JSON
        echo -e "  ${YELLOW}[4] ${PAIR}_${SHAPE}: neuron-profile view...${NC}"
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
    rm -rf /tmp/neuron_indep_${PAIR}_*
    echo ""
done

# 결과 요약
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Independent 프로파일링 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "출력 디렉토리: ${OUTPUT_DIR}"
echo ""
echo -e "${YELLOW}생성된 파일 목록:${NC}"
ls -lh "${OUTPUT_DIR}/" 2>/dev/null || echo "파일이 없습니다"
echo ""
echo -e "${GREEN}총 ${TOTAL_JSON}개의 JSON 프로파일 생성됨${NC}"
