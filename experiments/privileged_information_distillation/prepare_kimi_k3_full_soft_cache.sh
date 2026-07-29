#!/bin/bash
# Combine the frozen varied and instructed Kimi K3 binary-soft caches.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
VARIED_ROOT="results/blackbox/kimi_k3_fireworks_nothink_tvg_binary_logit_v1/train"
INSTRUCTED_ROOT="results/blackbox/kimi_k3_fireworks_nothink_tvg_binary_logit_instructed_v1/train"
FULL_ROOT="results/blackbox/kimi_k3_fireworks_nothink_tvg_binary_logit_full_v1/train"

sha256sum -c "${VARIED_ROOT}/SHA256SUMS"
sha256sum -c "${INSTRUCTED_ROOT}/SHA256SUMS"
mkdir -p "${FULL_ROOT}"

"${PYTHON}" \
  experiments/privileged_information_distillation/build_soft_teacher_cache.py \
  "${VARIED_ROOT}/generations.jsonl" \
  "${FULL_ROOT}/soft_targets.jsonl" \
  --additional-input "${INSTRUCTED_ROOT}/generations.jsonl" \
  --kind binary_identity \
  --expected-rows 6573

"${PYTHON}" \
  experiments/privileged_information_distillation/build_soft_student_rows.py \
  --soft-targets "${FULL_ROOT}/soft_targets.jsonl" \
  --output "${FULL_ROOT}/student_rows.jsonl" \
  --config-name pid_kimi_k3_openrouter_tvg_binary_soft_full_r16_ep2_v1 \
  --split train \
  --dataset-name-contains "" \
  --expected-rows 6573 \
  --source public_split_plus_kimi_k3_openrouter_full_binary_soft_target

sha256sum \
  "${FULL_ROOT}/soft_targets.jsonl" \
  "${FULL_ROOT}/student_rows.jsonl" \
  > "${FULL_ROOT}/SHA256SUMS"

sha256sum -c "${FULL_ROOT}/SHA256SUMS"
