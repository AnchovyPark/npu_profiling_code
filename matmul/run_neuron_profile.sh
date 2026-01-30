#!/bin/bash
# Neuron Profiling Script (Bucketing Version)
# bucketing_profile_v3.py 실행 -> NEFF 파일 찾기 -> NTFF 캡처 -> JSON 추출

set -e  # 에러 발생 시 스크립트 중단

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 설정 변수
OUTPUT_DIR="./neuron_profile_results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python 스크립트 직접 지정 (환경변수 또는 커맨드라인 인자)
PYTHON_SCRIPT=${1:-${PYTHON_SCRIPT:-"bucketing_profile_v4.py"}}
SEARCH_PATTERN="neuron_bucket_*"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Neuron Profile 자동화 스크립트${NC}"
echo -e "${GREEN}Script: ${PYTHON_SCRIPT}${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Step 0: 기존 NEFF 파일 정리
echo -e "${YELLOW}[0/4] 기존 NEFF 파일 정리 중...${NC}"
NEFF_DIRS_TO_DELETE=$(find /tmp -maxdepth 1 -type d -name "neuron_*" 2>/dev/null)
if [ -n "$NEFF_DIRS_TO_DELETE" ]; then
    NEFF_DELETE_COUNT=$(echo "$NEFF_DIRS_TO_DELETE" | wc -l)
    echo -e "  발견된 기존 NEFF 디렉토리: ${NEFF_DELETE_COUNT}개"
    rm -rf /tmp/neuron_*
    echo -e "${GREEN}[0/4] 완료: 기존 NEFF 파일 삭제됨${NC}"
else
    echo -e "${GREEN}[0/4] 완료: 삭제할 NEFF 파일 없음${NC}"
fi
echo ""

# Step 1: Python 스크립트 실행하여 NEFF 생성
echo -e "${YELLOW}[1/4] ${PYTHON_SCRIPT} 실행 중...${NC}"
python3 "${SCRIPT_DIR}/${PYTHON_SCRIPT}"
echo -e "${GREEN}[1/4] 완료: ${PYTHON_SCRIPT} 실행됨${NC}"
echo ""

# Step 2: 생성된 NEFF 파일들 찾기
echo -e "${YELLOW}[2/4] NEFF 파일 검색 중...${NC}"
NEFF_DIRS=$(find /tmp -maxdepth 1 -type d -name "${SEARCH_PATTERN}" 2>/dev/null | sort)

if [ -z "$NEFF_DIRS" ]; then
    echo -e "${RED}오류: ${SEARCH_PATTERN} 디렉토리를 찾을 수 없습니다${NC}"
    echo -e "${RED}${PYTHON_SCRIPT}가 정상적으로 실행되었는지 확인하세요${NC}"
    exit 1
fi

NEFF_COUNT=$(echo "$NEFF_DIRS" | wc -l)
echo -e "${GREEN}${NEFF_COUNT}개의 컴파일 디렉토리 발견${NC}"
echo ""

# 출력 디렉토리 생성
mkdir -p ${OUTPUT_DIR}

# Step 3 & 4: 각 NEFF 파일에 대해 프로파일링 수행
CURRENT=0
for COMPILER_DIR in $NEFF_DIRS; do
    CURRENT=$((CURRENT + 1))

    # 디렉토리 이름에서 M, K, N 추출
    DIR_NAME=$(basename "$COMPILER_DIR")
    MATRIX_SIZE=$(echo "$DIR_NAME" | sed 's/neuron_matmul_//')

    echo -e "${BLUE}[${CURRENT}/${NEFF_COUNT}] 처리 중: ${MATRIX_SIZE}${NC}"

    # NEFF 파일 찾기
    NEFF_FILE=$(find "$COMPILER_DIR" -name "*.neff" -type f | head -n 1)

    if [ -z "$NEFF_FILE" ]; then
        echo -e "${RED}  경고: ${COMPILER_DIR}에서 NEFF 파일을 찾을 수 없습니다. 건너뜁니다.${NC}"
        echo ""
        continue
    fi

    echo -e "  NEFF: ${NEFF_FILE}"

    # 출력 파일 경로 설정
    NTFF_FILE="${OUTPUT_DIR}/profile_${MATRIX_SIZE}.ntff"
    JSON_FILE="${OUTPUT_DIR}/profile_${MATRIX_SIZE}.json"

    # neuron-profile capture로 NTFF 생성
    echo -e "  ${YELLOW}[3/4] neuron-profile capture 실행 중...${NC}"
    neuron-profile capture \
        -n "${NEFF_FILE}" \
        -s "${NTFF_FILE}" 2>&1 | grep -v "level=info" || true

    if [ ! -f "${NTFF_FILE}" ]; then
        echo -e "${RED}  경고: NTFF 파일 생성 실패. 건너뜁니다.${NC}"
        echo ""
        continue
    fi

    echo -e "  ${GREEN}NTFF 생성 완료${NC}"

    # neuron-profile view로 JSON 추출
    echo -e "  ${YELLOW}[4/4] neuron-profile view 실행 중...${NC}"
    neuron-profile view \
        -n "${NEFF_FILE}" \
        -s "${NTFF_FILE}" \
        --output-format json \
        --output-file "${JSON_FILE}" 2>&1 | grep -v "level=info" || true

    if [ ! -f "${JSON_FILE}" ]; then
        echo -e "${RED}  경고: JSON 파일 생성 실패${NC}"
        echo ""
        continue
    fi

    echo -e "  ${GREEN}JSON 생성 완료: ${JSON_FILE}${NC}"
    echo -e "  파일 크기: $(du -h ${JSON_FILE} | cut -f1)"
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
ls -lh ${OUTPUT_DIR}/ 2>/dev/null || echo "파일이 없습니다"
echo ""

# 생성된 JSON 파일 개수 표시
JSON_COUNT=$(find ${OUTPUT_DIR} -name "*.json" -type f 2>/dev/null | wc -l)
echo -e "${GREEN}총 ${JSON_COUNT}개의 JSON 프로파일 생성됨${NC}"
