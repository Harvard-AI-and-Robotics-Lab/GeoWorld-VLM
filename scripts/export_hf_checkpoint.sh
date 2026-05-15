#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${REPO_ROOT}/configs/paths.local.env"

CKPT="${CKPT:?Set CKPT to the trainable_state.pt path}"
EXPORT_DIR="${EXPORT_DIR:?Set EXPORT_DIR to the output Hugging Face model directory}"
GPUS="${GPUS:-0}"

cd "${REPO_ROOT}/code/training"

CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONPATH="${REPO_ROOT}/code:${REPO_ROOT}/code/training:${PYTHONPATH:-}" \
python export_hf_model.py \
  --ckpt "${CKPT}" \
  --base-model "${GEMMA_MODEL}" \
  --output "${EXPORT_DIR}"

