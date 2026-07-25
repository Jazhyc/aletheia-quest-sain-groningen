#!/bin/bash
# Train and gate a rank-1 HP-KR specialist on the persistent Lambda H100.

set -euo pipefail

PHASE="${1:-}"
if [[ -z "${PHASE}" ]]; then
  echo "usage: $0 train|development|select|confirmation|confirm" >&2
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
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

METHOD="liars_bench_hpkr_rank1_specialist_v1"
ARTIFACT_ROOT="results/blackbox/${METHOD}"
EVAL_ARTIFACT="results/blackbox/liars_bench_ood_gptoss_v1/eval.jsonl"
TEACHER_ARTIFACT="results/blackbox/liars_bench_ood_gptoss_v1/teacher/train.jsonl"
PHOENIX_ADAPTER="results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter"
PHOENIX_PROMPT="configs/liars_bench_prompt_honest_alternative.yaml"
LOG_DIR="logs/lambda/${METHOD}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${PHASE}-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

NAMES=(
  "r1_e1_lr5e5"
  "r1_e3_lr5e5"
  "r1_e3_lr1e4"
)
CONFIGS=(
  "pid_liars_bench_hpkr_specialist_rank1_e1_lr5e5_v1"
  "pid_liars_bench_hpkr_specialist_rank1_e3_lr5e5_v1"
  "pid_liars_bench_hpkr_specialist_rank1_e3_lr1e4_v1"
)
ADAPTERS=(
  "results/blackbox/qwen9b_pid_liars_hpkr_specialist_rank1_e1_lr5e5_v1/adapter"
  "results/blackbox/qwen9b_pid_liars_hpkr_specialist_rank1_e3_lr5e5_v1/adapter"
  "results/blackbox/qwen9b_pid_liars_hpkr_specialist_rank1_e3_lr1e4_v1/adapter"
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

selected_candidate() {
  python - "$1" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1]))["selected"]
if value is not None:
    print(value)
PY
}

adapter_for_name() {
  local wanted="$1"
  local offset
  for offset in "${!NAMES[@]}"; do
    if [[ "${NAMES[$offset]}" == "${wanted}" ]]; then
      echo "${ADAPTERS[$offset]}"
      return
    fi
  done
  echo "unknown candidate: ${wanted}" >&2
  exit 1
}

case "${PHASE}" in
  train)
    require_paths "${TEACHER_ARTIFACT}"
    for config in "${CONFIGS[@]}"; do
      python experiments/privileged_information_distillation/train_student_sft.py \
        --config-name "${config}"
    done
    ;;
  development)
    require_paths \
      "${EVAL_ARTIFACT}" \
      "${PHOENIX_ADAPTER}/adapter_config.json" \
      "${PHOENIX_PROMPT}"
    adapter_args=()
    for offset in "${!NAMES[@]}"; do
      require_paths "${ADAPTERS[$offset]}/adapter_config.json"
      adapter_args+=(--adapter "${NAMES[$offset]}=${ADAPTERS[$offset]}")
    done
    python experiments/liars_bench_distillation/evaluate_hpkr_rank1_specialist.py \
      --eval-artifact "${EVAL_ARTIFACT}" \
      --output-dir "${ARTIFACT_ROOT}/development" \
      --phoenix-adapter "${PHOENIX_ADAPTER}" \
      --phoenix-prompt-config "${PHOENIX_PROMPT}" \
      --split development \
      "${adapter_args[@]}"
    ;;
  select)
    require_paths "${ARTIFACT_ROOT}/development/result.json"
    candidate_args=()
    for name in "${NAMES[@]}"; do
      candidate_args+=(--candidate "${name}")
    done
    python experiments/liars_bench_distillation/select_hpkr_rank1_specialist.py \
      --result "${ARTIFACT_ROOT}/development/result.json" \
      --generation-dir "${ARTIFACT_ROOT}/development" \
      --minimum-gain 0.02 \
      --minimum-source-delta -0.05 \
      --maximum-parse-error-increase 5 \
      --output "${ARTIFACT_ROOT}/selection-development.json" \
      "${candidate_args[@]}"
    ;;
  confirmation)
    require_paths "${ARTIFACT_ROOT}/selection-development.json"
    SELECTED="$(selected_candidate "${ARTIFACT_ROOT}/selection-development.json")"
    if [[ -z "${SELECTED}" ]]; then
      echo "no rank-1 specialist passed development; confirmation is skipped"
      exit 0
    fi
    ADAPTER="$(adapter_for_name "${SELECTED}")"
    require_paths \
      "${ADAPTER}/adapter_config.json" \
      "${PHOENIX_ADAPTER}/adapter_config.json"
    python experiments/liars_bench_distillation/evaluate_hpkr_rank1_specialist.py \
      --eval-artifact "${EVAL_ARTIFACT}" \
      --output-dir "${ARTIFACT_ROOT}/confirmation" \
      --phoenix-adapter "${PHOENIX_ADAPTER}" \
      --phoenix-prompt-config "${PHOENIX_PROMPT}" \
      --split confirmation \
      --adapter "${SELECTED}=${ADAPTER}"
    ;;
  confirm)
    require_paths \
      "${ARTIFACT_ROOT}/selection-development.json" \
      "${ARTIFACT_ROOT}/confirmation/result.json"
    SELECTED="$(selected_candidate "${ARTIFACT_ROOT}/selection-development.json")"
    python experiments/liars_bench_distillation/select_hpkr_rank1_specialist.py \
      --result "${ARTIFACT_ROOT}/confirmation/result.json" \
      --generation-dir "${ARTIFACT_ROOT}/confirmation" \
      --minimum-gain 0.01 \
      --minimum-source-delta -0.05 \
      --maximum-parse-error-increase 5 \
      --candidate "${SELECTED}" \
      --output "${ARTIFACT_ROOT}/selection-confirmation.json"
    ;;
  *)
    echo "unknown phase: ${PHASE}" >&2
    exit 2
    ;;
esac
