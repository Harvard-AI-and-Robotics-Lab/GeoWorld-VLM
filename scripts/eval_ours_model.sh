#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${REPO_ROOT}/configs/paths.local.env"

OURS_MODEL="${OURS_MODEL:?Set OURS_MODEL to the exported Hugging Face model directory}"
MODEL_PATH="${OURS_MODEL}" "${REPO_ROOT}/scripts/eval_raw_model.sh"

