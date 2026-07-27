#!/bin/bash
# Cache varied-only binary TVG supervision from Qwen3.5-397B on Lambda.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ -f "${HOME}/.config/aletheia/runtime.env" ]]; then
  source "${HOME}/.config/aletheia/runtime.env"
fi
if [[ -f "${HOME}/.config/aletheia/secrets.env" ]]; then
  source "${HOME}/.config/aletheia/secrets.env"
fi

METHOD="qwen35_397b_fp8_nothink_truth_value_binary_logit_v1"
RUN_DIR="results/blackbox/${METHOD}/train"
LOG_DIR="logs/lambda/${METHOD}"
PYTHON="${ROOT}/.venv/bin/python"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/teacher.log") 2>&1

"${PYTHON}" experiments/blackbox/run_judge.py \
  --config-path ../../configs/single_judges \
  --config-name blackbox_reasoning_nothink_truth_value_binary_logit_qwen35_397b_fp8_v1 \
  split=train \
  dataset_name_contains=varied-deception

"${PYTHON}" experiments/privileged_information_distillation/build_soft_teacher_cache.py \
  "${RUN_DIR}/generations.jsonl" \
  "${RUN_DIR}/soft_targets.jsonl" \
  --kind binary_identity \
  --expected-rows 2880

"${PYTHON}" experiments/privileged_information_distillation/build_soft_student_rows.py \
  --soft-targets "${RUN_DIR}/soft_targets.jsonl" \
  --output "${RUN_DIR}/student_rows.jsonl" \
  --config-name pid_qwen397_tvg_binary_soft_distillation_v1 \
  --split train \
  --dataset-name-contains varied-deception \
  --expected-rows 2880

sha256sum \
  "${RUN_DIR}/generations.jsonl" \
  "${RUN_DIR}/soft_targets.jsonl" \
  "${RUN_DIR}/student_rows.jsonl" \
  > "${RUN_DIR}/SHA256SUMS"

du -h \
  "${RUN_DIR}/generations.jsonl" \
  "${RUN_DIR}/soft_targets.jsonl" \
  "${RUN_DIR}/student_rows.jsonl"
