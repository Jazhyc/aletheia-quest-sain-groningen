#!/bin/bash
# Cache Kimi K3 binary-margin supervision for instructed-deception training rows.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

METHOD="kimi_k3_fireworks_nothink_tvg_binary_logit_instructed_v1"
RUN_DIR="results/blackbox/${METHOD}/train"
PYTHON="${ROOT}/.venv/bin/python"
CONCURRENCY="${KIMI_CONCURRENCY:-64}"

"${PYTHON}" -m experiments.openrouter_qwen397_tvg.run_openrouter_tvg \
  --model moonshotai/kimi-k3 \
  --method "${METHOD}" \
  --split train \
  --dataset-name-contains instructed-deception \
  --concurrency "${CONCURRENCY}" \
  --max-retries 8 \
  --provider-only Fireworks \
  --no-allow-fallbacks

"${PYTHON}" \
  experiments/privileged_information_distillation/build_soft_teacher_cache.py \
  "${RUN_DIR}/generations.jsonl" \
  "${RUN_DIR}/soft_targets.jsonl" \
  --kind binary_identity \
  --expected-rows 3693

"${PYTHON}" \
  experiments/privileged_information_distillation/build_soft_student_rows.py \
  --soft-targets "${RUN_DIR}/soft_targets.jsonl" \
  --output "${RUN_DIR}/student_rows.jsonl" \
  --config-name pid_kimi_k3_openrouter_tvg_binary_soft_distillation_v1 \
  --split train \
  --dataset-name-contains instructed-deception \
  --expected-rows 3693 \
  --source public_split_plus_kimi_k3_openrouter_instructed_binary_soft_target

sha256sum \
  "${RUN_DIR}/generations.jsonl" \
  "${RUN_DIR}/soft_targets.jsonl" \
  "${RUN_DIR}/student_rows.jsonl" \
  > "${RUN_DIR}/SHA256SUMS"

sha256sum -c "${RUN_DIR}/SHA256SUMS"
