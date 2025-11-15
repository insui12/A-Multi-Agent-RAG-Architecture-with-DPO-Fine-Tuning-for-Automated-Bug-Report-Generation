#!/bin/bash
set -e

# =================================================================
# 1. 글로벌 설정 (DPO 파이프라인 - HF-Only)
# =================================================================

# --- [A] 실행할 단계 설정 (true / false) ---

# (1단계) SFT 학습을 실행합니다. (controller_agent.py --train_sft)
DO_TRAIN_SFT=true

# (2단계) data_agent.py를 실행하여 DPO 학습용 .jsonl 파일을 생성/캐시합니다.
# (SFT 학습이 완료되어야 합니다.)
DO_BUILD_DPO_DATA=true

# (3단계) DPO 학습을 실행합니다. (controller_agent.py --train_dpo)
# (1단계와 2단계가 먼저 완료되어야 합니다.)
DO_TRAIN_DPO=true

# (4단계) 최종 성능을 평가합니다. (controller_agent.py --adapter <type>)
DO_INFERENCE=true


# --- [B] 핵심 파라미터 ---

# (1, 2, 3, 4단계) 학습/추론/RAG에 사용할 프롬프트 컴포넌트
# (e.g., "base" "cot" "xai")
# [MODIFIED] 프롬프트 로직이 "base"로 고정됨. 이 변수는 현재 사용되지 않음.
PROMPT_COMPONENTS=("base") 

# (4단계) 최종 추론 시 테스트할 어댑터 타입:
# 'base' (어댑터 없음), 'sft' (SFT 어댑터), 'dpo' (DPO 어댑터)
INFER_ADAPTER_TYPE="dpo"

# (4단계) 최종 추론 시 RAG 퓨샷(k) 개수 (0, 1, 2, 3)
INFER_FEW_SHOT_K=1


# --- [C] 공통 설정 ---

# (모든 단계) 실행할 모델 (배열)
MODELS=("qwen")

# (모든 단계) 'qlora' (4비트) 또는 'lora' (16비트)
LORA_TYPE="lora"

# (2, 4단계) 배치 크기
BATCH_SIZE=4

# (4단계) 추론 시 샘플링 대신 greedy 사용
DO_GREEDY=true

# (모든 단계) 데이터 일부만 사용하는 테스트 실행
DO_TEST_RUN=false

# (모든 단계) 데이터셋 CSV 강제 재생성
DO_FORCE_RESPLIT=false


# =================================================================
# 2. 플래그 설정 (수정 불필요)
# =================================================================

TEST_FLAG=""
[ "$DO_TEST_RUN" = true ] && TEST_FLAG="--test_run"

GREEDY_FLAG=""
[ "$DO_GREEDY" = true ] && GREEDY_FLAG="--greedy"

FORCE_RESPLIT_FLAG=""
[ "$DO_FORCE_RESPLIT" = true ] && FORCE_RESPLIT_FLAG="--force_resplit"


# =================================================================
# 3. 메인 실행 루프
# =================================================================

echo "▶ Starting DPO Pipeline (HF-Only Mode)..."

# --- (1단계) SFT 학습 ---
if [ "$DO_TRAIN_SFT" = true ]; then
    echo "================================================================="
    echo "▶ (1/4) Executing: SFT Training (Controller)"
    echo "================================================================="
    
    python3 controller_agent.py \
        --train_sft \
        --models "${MODELS[@]}" \
        --lora_type "${LORA_TYPE}" \
        ${TEST_FLAG} \
        ${FORCE_RESPLIT_FLAG}

    echo "▶ (1/4) SFT Training Complete."
else
    echo "▶ (1/4) Skipping: SFT Training."
fi


# --- (2단계) DPO 데이터셋 구축 ---
if [ "$DO_BUILD_DPO_DATA" = true ]; then
    echo "================================================================="
    echo "▶ (2/4) Executing: Build DPO Dataset (Requires SFT Adapter)"
    echo "================================================================="
    
    # [MODIFIED] build_dpo_dataset.py -> data_agent.py로 변경
    # [MODIFIED] ${TEST_FLAG} 제거 (data_agent.py의 main은 이 인수를 받지 않음)
    python3 data_agent.py \
        --models "${MODELS[@]}" \
        --batch_size "${BATCH_SIZE}"

    echo "▶ (2/4) DPO Dataset Build Complete."
else
    echo "▶ (2/4) Skipping: DPO Dataset."
fi


# --- (3DPO 학습 ---
if [ "$DO_TRAIN_DPO" = true ]; then
    echo "================================================================="
    echo "▶ (3/4) Executing: DPO Training (Controller)"
    echo "================================================================="
    
    # controller_agent.py는 DPO 데이터셋이 존재한다고 가정하고 학습을 실행합니다.
    python3 controller_agent.py \
        --train_dpo \
        --models "${MODELS[@]}" \
        --lora_type "${LORA_TYPE}" \
        ${TEST_FLAG} \
        ${FORCE_RESPLIT_FLAG}

    echo "▶ (3/4) DPO Training Complete."
else
    echo "▶ (3/4) Skipping: DPO Training."
fi


# --- (4단계) 최종 추론 ---
if [ "$DO_INFERENCE" = true ]; then
    echo "================================================================="
    echo "▶ (4/4) Executing: Final Inference-Only"
    echo "▶ Adapter: ${INFER_ADAPTER_TYPE}"
    echo "▶ Few-Shot k: ${INFER_FEW_SHOT_K}"
    echo "================================================================="
    
    python3 controller_agent.py \
        --models "${MODELS[@]}" \
        --lora_type "${LORA_TYPE}" \
        --batch_size "${BATCH_SIZE}" \
        --adapter "${INFER_ADAPTER_TYPE}" \
        --few_shot_k "${INFER_FEW_SHOT_K}" \
        ${TEST_FLAG} \
        ${GREEDY_FLAG} \
        ${FORCE_RESPLIT_FLAG}

    echo "▶ (4/4) Final Inference Complete."
else
    echo "▶ (4/4) Skipping: Final Inference."
fi

echo "================================================================="
echo "▶▶ All Pipeline Steps Complete."
echo "================================================================="