#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${REPO_ROOT}/configs/paths.local.env"

TASK="${TASK:-whatsup_vsr}"
MODEL_PATH="${MODEL_PATH:-${GEMMA_MODEL}}"
GPUS="${GPUS:-0}"
mkdir -p "${RESULTS_DIR}"

if [[ "${TASK}" == "whatsup_vsr" ]]; then
  PYTHONPATH="${REPO_ROOT}/code:${REPO_ROOT}/code/training:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPUS}" \
  python "${REPO_ROOT}/code/evaluation/eval_gemma.py" \
    --model-path "${MODEL_PATH}" \
    --data-dir "${ADAPTVIS_DATA_DIR}" \
    --prompts-dir "${ADAPTVIS_PROMPTS_DIR}" \
    --split-file "${SPLIT_FILE:-${REPO_ROOT}/splits/data_split.json}" \
    --output-dir "${RESULTS_DIR}/raw_whatsup_vsr"
elif [[ "${TASK}" == "embspatial" ]]; then
  PYTHONPATH="${REPO_ROOT}/code:${REPO_ROOT}/code/training:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPUS}" \
  python "${REPO_ROOT}/code/evaluation/eval_embspatial_gemma.py" \
    --model-path "${MODEL_PATH}" \
    --json-path "${EMBSPATIAL_JSON}" \
    --split-file "${SPLIT_FILE:-${REPO_ROOT}/splits/embspatial_split.json}" \
    --output-dir "${RESULTS_DIR}/raw_embspatial"
elif [[ "${TASK}" == "sat" ]]; then
  PYTHONPATH="${REPO_ROOT}/code:${REPO_ROOT}/code/training:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPUS}" \
  python "${REPO_ROOT}/code/evaluation/eval_sat_gemma_qtype.py" \
    --model-path "${MODEL_PATH}" \
    --sat-root "${SAT_ROOT}" \
    --split-file "${SPLIT_FILE:-${REPO_ROOT}/splits/sat_split.json}" \
    --sat_qtype_filter "${SAT_QTYPE_FILTER:-all}" \
    --output-dir "${RESULTS_DIR}/raw_sat"
else
  echo "Unknown TASK=${TASK}. Use whatsup_vsr, embspatial, or sat." >&2
  exit 1
fi

