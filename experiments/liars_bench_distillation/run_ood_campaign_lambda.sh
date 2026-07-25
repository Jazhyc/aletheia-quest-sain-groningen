#!/bin/bash
# Run one phase of the GPT-OSS Liars' Bench OOD campaign on Lambda.

set -euo pipefail

PHASE="${1:-}"
if [[ -z "${PHASE}" ]]; then
  echo "usage: $0 teacher|train|external-eval|validation|analyze" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ -f "${HOME}/.config/aletheia/runtime.env" ]]; then
  source "${HOME}/.config/aletheia/runtime.env"
fi
if [[ -f "${HOME}/.config/aletheia/secrets.env" ]]; then
  source "${HOME}/.config/aletheia/secrets.env"
fi
source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

METHOD="liars_bench_ood_gptoss_v1"
ARTIFACT_ROOT="results/blackbox/${METHOD}"
LOG_DIR="logs/lambda/${METHOD}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${PHASE}-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

PARENT_ADAPTER="results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter"
CANDIDATE_ADAPTER="results/blackbox/qwen9b_pid_liars_ood_gptoss_replay_continue_adamw2e5_v1/adapter"
VALIDATION_RUN="validation_liars_ood_gptoss_v1"
TEACHER_SERVER_PID=""

EXCLUSIONS=(
  "results/blackbox/liars_bench_pid_aug_v1/teacher/train.jsonl"
  "results/blackbox/liars_bench_pid_aug_v1/eval.jsonl"
  "results/blackbox/liars_bench_frozen_judge_signatures_v1/truth.jsonl"
  "results/blackbox/liars_bench_heavy_spectrum_confirmation_v1/eval.jsonl"
  "results/blackbox/liars_bench_soft_trigger_gemma_confirmation_v1/eval.jsonl"
  "results/blackbox/liars_bench_passage_true_false_v1/eval.jsonl"
)

require_paths() {
  local path
  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      echo "required path is missing: ${path}" >&2
      exit 1
    fi
  done
}

run_teacher() {
  require_paths "${EXCLUSIONS[@]}"
  local model="openai/gpt-oss-120b"
  local served_model="gpt-oss-teacher"
  local port="${VLLM_PORT:-18080}"
  local api_base="http://127.0.0.1:${port}/v1"
  local server_log="${LOG_DIR}/teacher-server-$(date -u +%Y%m%dT%H%M%SZ).out"

  vllm serve "${model}" \
    --served-model-name "${served_model}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    >"${server_log}" 2>&1 &
  TEACHER_SERVER_PID=$!
  cleanup_teacher() {
    if [[ -n "${TEACHER_SERVER_PID}" ]]; then
      kill "${TEACHER_SERVER_PID}" 2>/dev/null || true
      wait "${TEACHER_SERVER_PID}" 2>/dev/null || true
    fi
  }
  trap cleanup_teacher EXIT

  local attempt
  for attempt in $(seq 1 300); do
    if curl -fsS "${api_base}/models" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${TEACHER_SERVER_PID}" 2>/dev/null; then
      echo "GPT-OSS server exited during startup; see ${server_log}" >&2
      exit 1
    fi
    sleep 2
  done
  curl -fsS "${api_base}/models" >/dev/null

  local exclusion_args=()
  local path
  for path in "${EXCLUSIONS[@]}"; do
    exclusion_args+=(--exclude-artifact "${path}")
  done
  python experiments/liars_bench_distillation/prepare_teacher_data.py \
    --api-base "${api_base}" \
    --served-model "${served_model}" \
    --artifact "${ARTIFACT_ROOT}/teacher/train.jsonl" \
    --eval-artifact "${ARTIFACT_ROOT}/eval.jsonl" \
    --per-label-train 128 \
    --per-label-eval 100 \
    --seed 20260725 \
    --reasoning-effort medium \
    "${exclusion_args[@]}"

  python experiments/liars_bench_distillation/audit_teacher_cache.py \
    "${ARTIFACT_ROOT}/teacher/train.jsonl" \
    --output "${ARTIFACT_ROOT}/teacher/audit.json" \
    --expected-total 1024 \
    --expected-datasets 4
}

case "${PHASE}" in
  teacher)
    run_teacher
    ;;
  train)
    require_paths \
      "${ARTIFACT_ROOT}/teacher/train.jsonl" \
      "${PARENT_ADAPTER}/adapter_config.json" \
      "results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl"
    python experiments/privileged_information_distillation/train_student_sft.py \
      --config-name pid_liars_bench_ood_gptoss_continue_v1
    ;;
  external-eval)
    require_paths \
      "${ARTIFACT_ROOT}/eval.jsonl" \
      "${PARENT_ADAPTER}/adapter_config.json" \
      "${CANDIDATE_ADAPTER}/adapter_config.json"
    python experiments/liars_bench_distillation/evaluate_students.py \
      --eval-artifact "${ARTIFACT_ROOT}/eval.jsonl" \
      --output-dir "${ARTIFACT_ROOT}/evaluation" \
      --adapter "baseline=${PARENT_ADAPTER}" \
      --adapter "ood_gptoss=${CANDIDATE_ADAPTER}"
    ;;
  validation)
    require_paths \
      "${PARENT_ADAPTER}/adapter_config.json" \
      "${CANDIDATE_ADAPTER}/adapter_config.json" \
      "dev_splits/validation.csv"
    python experiments/privileged_information_distillation/evaluate_student_sft.py \
      --adapter-dir "${PARENT_ADAPTER}" \
      --adapter-dir "${CANDIDATE_ADAPTER}" \
      --split validation \
      --run-name "${VALIDATION_RUN}"
    ;;
  analyze)
    require_paths \
      "${PARENT_ADAPTER}/${VALIDATION_RUN}/result.json" \
      "${CANDIDATE_ADAPTER}/${VALIDATION_RUN}/result.json" \
      "${ARTIFACT_ROOT}/evaluation/result.json"
    python experiments/liars_bench_distillation/analyze_transfer.py \
      --competition "baseline=${PARENT_ADAPTER}/${VALIDATION_RUN}/result.json" \
      --competition "ood_gptoss=${CANDIDATE_ADAPTER}/${VALIDATION_RUN}/result.json" \
      --external-result "${ARTIFACT_ROOT}/evaluation/result.json" \
      --max-competition-loss 0.0025 \
      --min-external-gain 0.02 \
      --minimum-category-delta -0.02 \
      --minimum-category-source-model-delta -0.05 \
      --output "${ARTIFACT_ROOT}/analysis.json"
    ;;
  *)
    echo "unknown phase: ${PHASE}" >&2
    exit 2
    ;;
esac
