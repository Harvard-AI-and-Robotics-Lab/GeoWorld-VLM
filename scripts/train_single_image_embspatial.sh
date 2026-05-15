#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${REPO_ROOT}/configs/paths.local.env"

SPLIT_FILE="${SPLIT_FILE:-${REPO_ROOT}/splits/embspatial_split.json}"
EXP_NAME="${EXP_NAME:-gemma4_lingbot_embspatial}"
GPUS="${GPUS:-0,1}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}/code/training"

CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONPATH="${REPO_ROOT}/code:${REPO_ROOT}/code/training:${PYTHONPATH:-}" \
python train_embspatial_gemma4_lingbot_spatial.py \
  --gpu 0 --teacher_gpu 1 \
  --gemma_model_name "${GEMMA_MODEL}" \
  --lingbot_model_dir "${LINGBOT_MODEL}" \
  --lingbot_code_dir "${LINGBOT_CODE}" \
  --json_path "${EMBSPATIAL_JSON}" \
  --split_file "${SPLIT_FILE}" \
  --save_dir "${OUTPUT_DIR}/${EXP_NAME}" \
  --epochs "${EPOCHS:-3}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-4}" \
  --lambda_align "${LAMBDA_ALIGN:-0.1}" \
  --lambda_preserve "${LAMBDA_PRESERVE:-0.05}" \
  --teacher_mode "${TEACHER_MODE:-i2v}" \
  --i2v_num_frames "${I2V_NUM_FRAMES:-9}" \
  --num_teacher_steps "${NUM_TEACHER_STEPS:-2}" \
  --use_fast_model \
  --wan_hook_block_index "${WAN_HOOK_BLOCK_INDEX:-24}" \
  --use_camera_perturbation \
  --wan_prompt_text "${WAN_PROMPT_TEXT:-A slight camera motion with stable object layout and unchanged spatial relations.}"

