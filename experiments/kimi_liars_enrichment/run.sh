#!/bin/bash
# Prepare, query, and package Kimi K3 Liars' Bench binary-soft supervision.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source .venv/bin/activate

LIARS_ROOT="${LIARS_ROOT:-/scratch/s4626451/.huggingface/hub/datasets--Cadenza-Labs--liars-bench/snapshots/65299c5b10aa07adf75716ecb875c6713eed0dde}"
PILOT="results/blackbox/liars_bench_pid_aug_v1/eval.jsonl"
RUN_ROOT="results/blackbox/kimi_k3_liars_semantic_soft_v1"
PILOT_ROOT="${RUN_ROOT}/pilot"
TRAIN_ROOT="${RUN_ROOT}/train"
PHASE="${1:-}"

case "${PHASE}" in
  pilot)
    python experiments/kimi_liars_enrichment/query.py \
      --input "${PILOT}" \
      --output "${PILOT_ROOT}/ordinary.jsonl" \
      --condition ordinary
    python experiments/kimi_liars_enrichment/query.py \
      --input "${PILOT}" \
      --output "${PILOT_ROOT}/semantic.jsonl" \
      --condition semantic
    python experiments/kimi_liars_enrichment/audit.py \
      --ordinary "${PILOT_ROOT}/ordinary.jsonl" \
      --semantic "${PILOT_ROOT}/semantic.jsonl" \
      --pilot-artifact "${PILOT}" \
      --output "${PILOT_ROOT}/audit.json"
    ;;
  prepare)
    python experiments/kimi_liars_enrichment/prepare.py \
      --liars-root "${LIARS_ROOT}" \
      --pilot-artifact "${PILOT}" \
      --audit "${PILOT_ROOT}/audit.json" \
      --output "${TRAIN_ROOT}/student_rows.jsonl"
    ;;
  query)
    python experiments/kimi_liars_enrichment/query.py \
      --input "${TRAIN_ROOT}/student_rows.jsonl" \
      --output "${TRAIN_ROOT}/generations.jsonl" \
      --condition selected
    ;;
  build)
    rows="$(wc -l < "${TRAIN_ROOT}/generations.jsonl")"
    python experiments/privileged_information_distillation/build_soft_teacher_cache.py \
      "${TRAIN_ROOT}/generations.jsonl" \
      "${TRAIN_ROOT}/soft_targets.jsonl" \
      --kind binary_identity \
      --expected-rows "${rows}"
    sha256sum \
      "${TRAIN_ROOT}/generations.jsonl" \
      "${TRAIN_ROOT}/soft_targets.jsonl" \
      "${TRAIN_ROOT}/student_rows.jsonl" \
      > "${TRAIN_ROOT}/SHA256SUMS"
    sha256sum -c "${TRAIN_ROOT}/SHA256SUMS"
    python experiments/kimi_liars_enrichment/audit_full_cache.py \
      --generations "${TRAIN_ROOT}/generations.jsonl" \
      --students "${TRAIN_ROOT}/student_rows.jsonl" \
      --output "${TRAIN_ROOT}/audit.json" \
      --expected-rows "${rows}"
    ;;
  *)
    echo "usage: $0 {pilot|prepare|query|build}" >&2
    exit 2
    ;;
esac
