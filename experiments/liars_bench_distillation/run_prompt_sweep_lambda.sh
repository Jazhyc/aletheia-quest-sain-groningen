#!/bin/bash
# Run the frozen Liars' Bench prompt sweep on the persistent Lambda H100.

set -euo pipefail

PHASE="${1:-}"
if [[ -z "${PHASE}" ]]; then
  echo "usage: $0 development|select|confirmation|confirm|validation|analyze" >&2
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

METHOD="liars_bench_prompt_sweep_v1"
ARTIFACT_ROOT="results/blackbox/${METHOD}"
LOG_DIR="logs/lambda/${METHOD}"
PARENT_ADAPTER="results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter"
PARENT_ROOT="${PARENT_ADAPTER%/adapter}"
EVAL_ARTIFACT="results/blackbox/liars_bench_ood_gptoss_v1/eval.jsonl"
VALIDATION_RUN="validation_liars_prompt_sweep_v1"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${PHASE}-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

CONTROL_CONFIG="configs/liars_bench_prompt_control.yaml"
MODE_CONFIG="configs/liars_bench_prompt_mode_first.yaml"
LEDGER_CONFIG="configs/liars_bench_prompt_claim_ledger.yaml"
ALTERNATIVE_CONFIG="configs/liars_bench_prompt_honest_alternative.yaml"

require_paths() {
  local path
  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      echo "required path is missing: ${path}" >&2
      exit 1
    fi
  done
}

selected_prompt() {
  local selection_path="$1"
  python - "${selection_path}" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1]))["selected"]
if value is not None:
    print(value)
PY
}

prompt_config() {
  case "$1" in
    mode_first) echo "${MODE_CONFIG}" ;;
    claim_ledger) echo "${LEDGER_CONFIG}" ;;
    honest_alternative) echo "${ALTERNATIVE_CONFIG}" ;;
    *)
      echo "unknown selected prompt: $1" >&2
      exit 1
      ;;
  esac
}

case "${PHASE}" in
  development)
    require_paths \
      "${EVAL_ARTIFACT}" \
      "${PARENT_ADAPTER}/adapter_config.json" \
      "${CONTROL_CONFIG}" \
      "${MODE_CONFIG}" \
      "${LEDGER_CONFIG}" \
      "${ALTERNATIVE_CONFIG}"
    python experiments/liars_bench_distillation/evaluate_prompt_sweep.py \
      --eval-artifact "${EVAL_ARTIFACT}" \
      --adapter-dir "${PARENT_ADAPTER}" \
      --output-dir "${ARTIFACT_ROOT}/development" \
      --split development \
      --expected-rows 400 \
      --prompt "control=${CONTROL_CONFIG}" \
      --prompt "mode_first=${MODE_CONFIG}" \
      --prompt "claim_ledger=${LEDGER_CONFIG}" \
      --prompt "honest_alternative=${ALTERNATIVE_CONFIG}"
    ;;
  select)
    require_paths "${ARTIFACT_ROOT}/development/result.json"
    python experiments/liars_bench_distillation/select_prompt_sweep.py \
      --result "${ARTIFACT_ROOT}/development/result.json" \
      --generation-dir "${ARTIFACT_ROOT}/development" \
      --minimum-macro-gain 0.03 \
      --minimum-category-delta -0.02 \
      --minimum-category-source-model-delta -0.05 \
      --maximum-parse-error-increase 10 \
      --output "${ARTIFACT_ROOT}/selection-development.json"
    ;;
  confirmation)
    require_paths \
      "${EVAL_ARTIFACT}" \
      "${PARENT_ADAPTER}/adapter_config.json" \
      "${ARTIFACT_ROOT}/selection-development.json"
    SELECTED="$(selected_prompt "${ARTIFACT_ROOT}/selection-development.json")"
    if [[ -z "${SELECTED}" ]]; then
      echo "no development prompt passed; confirmation is intentionally skipped"
      exit 0
    fi
    SELECTED_CONFIG="$(prompt_config "${SELECTED}")"
    python experiments/liars_bench_distillation/evaluate_prompt_sweep.py \
      --eval-artifact "${EVAL_ARTIFACT}" \
      --adapter-dir "${PARENT_ADAPTER}" \
      --output-dir "${ARTIFACT_ROOT}/confirmation" \
      --split confirmation \
      --expected-rows 400 \
      --prompt "control=${CONTROL_CONFIG}" \
      --prompt "${SELECTED}=${SELECTED_CONFIG}"
    ;;
  confirm)
    require_paths \
      "${ARTIFACT_ROOT}/selection-development.json" \
      "${ARTIFACT_ROOT}/confirmation/result.json"
    SELECTED="$(selected_prompt "${ARTIFACT_ROOT}/selection-development.json")"
    python experiments/liars_bench_distillation/select_prompt_sweep.py \
      --result "${ARTIFACT_ROOT}/confirmation/result.json" \
      --generation-dir "${ARTIFACT_ROOT}/confirmation" \
      --minimum-macro-gain 0.02 \
      --minimum-category-delta -0.02 \
      --minimum-category-source-model-delta -0.05 \
      --maximum-parse-error-increase 10 \
      --output "${ARTIFACT_ROOT}/selection-confirmation.json"
    CONFIRMED="$(selected_prompt "${ARTIFACT_ROOT}/selection-confirmation.json")"
    if [[ "${CONFIRMED}" != "${SELECTED}" ]]; then
      echo "development winner did not pass confirmation"
    fi
    ;;
  validation)
    require_paths \
      "${PARENT_ADAPTER}/adapter_config.json" \
      "${ARTIFACT_ROOT}/selection-confirmation.json" \
      "dev_splits/dry.validation.yaml"
    SELECTED="$(selected_prompt "${ARTIFACT_ROOT}/selection-confirmation.json")"
    if [[ -z "${SELECTED}" ]]; then
      echo "no prompt passed confirmation; validation is intentionally skipped"
      exit 0
    fi
    SELECTED_CONFIG="$(prompt_config "${SELECTED}")"
    python experiments/privileged_information_distillation/evaluate_student_sft.py \
      --adapter-dir "${PARENT_ADAPTER}" \
      --split validation \
      --run-name "${VALIDATION_RUN}" \
      --prompt-condition "control=${CONTROL_CONFIG}" \
      --prompt-condition "${SELECTED}=${SELECTED_CONFIG}"
    ;;
  analyze)
    require_paths "${ARTIFACT_ROOT}/selection-confirmation.json"
    SELECTED="$(selected_prompt "${ARTIFACT_ROOT}/selection-confirmation.json")"
    if [[ -z "${SELECTED}" ]]; then
      echo "no prompt passed confirmation; final validation analysis is skipped"
      exit 0
    fi
    require_paths \
      "${PARENT_ROOT}/${VALIDATION_RUN}/control/result.json" \
      "${PARENT_ROOT}/${VALIDATION_RUN}/${SELECTED}/result.json"
    python experiments/liars_bench_distillation/analyze_prompt_validation.py \
      --control-result "${PARENT_ROOT}/${VALIDATION_RUN}/control/result.json" \
      --candidate-result "${PARENT_ROOT}/${VALIDATION_RUN}/${SELECTED}/result.json" \
      --control-generations "${PARENT_ROOT}/${VALIDATION_RUN}/control/generations.jsonl" \
      --candidate-generations "${PARENT_ROOT}/${VALIDATION_RUN}/${SELECTED}/generations.jsonl" \
      --confirmation "${ARTIFACT_ROOT}/selection-confirmation.json" \
      --candidate-name "${SELECTED}" \
      --maximum-balanced-accuracy-loss 0.0025 \
      --maximum-scenario-loss 0.01 \
      --maximum-parse-error-increase 10 \
      --output "${ARTIFACT_ROOT}/analysis.json"
    ;;
  *)
    echo "unknown phase: ${PHASE}" >&2
    exit 2
    ;;
esac
