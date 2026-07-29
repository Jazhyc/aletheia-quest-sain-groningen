#!/bin/bash
# Cache regular Qwen3.5-397B clarified-prompt supervision through OpenRouter.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

METHOD="qwen35_397b_openrouter_nothink_tvg_binary_logit_explicit_digits_v1"
RUN_DIR="results/blackbox/${METHOD}/train"
PYTHON="${ROOT}/.venv/bin/python"

"${PYTHON}" -m experiments.openrouter_qwen397_tvg.run_openrouter_tvg \
  --model qwen/qwen3.5-397b-a17b \
  --method "${METHOD}" \
  --prompt-variant explicit_digits \
  --split train \
  --dataset-name-contains varied-deception \
  --concurrency 8 \
  --max-retries 8 \
  --provider-only Alibaba \
  --no-allow-fallbacks

"${PYTHON}" \
  experiments/privileged_information_distillation/build_soft_teacher_cache.py \
  "${RUN_DIR}/generations.jsonl" \
  "${RUN_DIR}/soft_targets.jsonl" \
  --kind binary_identity \
  --expected-rows 2880

"${PYTHON}" \
  experiments/privileged_information_distillation/build_soft_student_rows.py \
  --soft-targets "${RUN_DIR}/soft_targets.jsonl" \
  --output "${RUN_DIR}/student_rows.jsonl" \
  --config-name pid_qwen397_openrouter_explicit_tvg_binary_soft_distillation_v1 \
  --split train \
  --dataset-name-contains varied-deception \
  --expected-rows 2880 \
  --source public_split_plus_qwen397_openrouter_explicit_binary_soft_target

sha256sum \
  "${RUN_DIR}/generations.jsonl" \
  "${RUN_DIR}/soft_targets.jsonl" \
  "${RUN_DIR}/student_rows.jsonl" \
  > "${RUN_DIR}/SHA256SUMS"

sha256sum -c "${RUN_DIR}/SHA256SUMS"
