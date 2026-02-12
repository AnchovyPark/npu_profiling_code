#!/bin/bash
# matmul_diverse independent profiling

set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/neuron_profile_results"
mkdir -p "${OUTPUT_DIR}"

SCRIPTS=$(ls "${SCRIPT_DIR}"/*.py 2>/dev/null)
TOTAL_JSON=0

for SCRIPT in $SCRIPTS; do
    NAME=$(basename "$SCRIPT" .py)
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  ${NAME} (independent)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    echo -e "${YELLOW}[0] 기존 NEFF 정리${NC}"
    rm -rf /tmp/neuron_indep_${NAME}_*

    echo -e "${YELLOW}[1] ${NAME}.py 실행 중...${NC}"
    python3 "${SCRIPT}"
    echo -e "${GREEN}[1] 완료${NC}"

    NEFF_DIRS=$(find /tmp -maxdepth 1 -type d -name "neuron_indep_${NAME}_*" 2>/dev/null | sort)
    if [ -z "$NEFF_DIRS" ]; then
        echo -e "${RED}  NEFF 디렉토리 없음${NC}"
        continue
    fi

    for COMPILER_DIR in $NEFF_DIRS; do
        DIR_NAME=$(basename "$COMPILER_DIR")
        SHAPE=$(echo "$DIR_NAME" | sed "s/neuron_indep_//")

        NEFF_FILE=$(find "$COMPILER_DIR" -name "*.neff" -type f | head -n 1)
        if [ -z "$NEFF_FILE" ]; then
            echo -e "  ${RED}NEFF 파일 없음${NC}"
            continue
        fi

        NTFF_FILE="${OUTPUT_DIR}/profile_${SHAPE}.ntff"
        JSON_FILE="${OUTPUT_DIR}/profile_${SHAPE}.json"

        echo -e "  ${YELLOW}[3] neuron-profile capture...${NC}"
        neuron-profile capture -n "${NEFF_FILE}" -s "${NTFF_FILE}" 2>&1 | grep -v "level=info" || true

        if [ ! -f "${NTFF_FILE}" ]; then
            echo -e "  ${RED}NTFF 생성 실패${NC}"
            continue
        fi

        echo -e "  ${YELLOW}[4] neuron-profile view...${NC}"
        neuron-profile view -n "${NEFF_FILE}" -s "${NTFF_FILE}" --output-format json --output-file "${JSON_FILE}" 2>&1 | grep -v "level=info" || true

        if [ -f "${JSON_FILE}" ]; then
            echo -e "  ${GREEN}OK: ${JSON_FILE} ($(du -h "${JSON_FILE}" | cut -f1))${NC}"
            TOTAL_JSON=$((TOTAL_JSON + 1))
        else
            echo -e "  ${RED}JSON 생성 실패${NC}"
        fi
    done

    rm -rf /tmp/neuron_indep_${NAME}_*
    echo ""
done

echo -e "${GREEN}완료! 총 ${TOTAL_JSON}개 JSON 생성${NC}"
ls -lh "${OUTPUT_DIR}/"
