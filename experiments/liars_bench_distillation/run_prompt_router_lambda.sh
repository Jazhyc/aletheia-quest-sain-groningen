#!/bin/bash
# Run the frozen semantic prompt-router experiment on the persistent Lambda H100.

set -euo pipefail

PHASE="${1:-}"
if [[ -z "${PHASE}" ]]; then
  echo "usage: $0 development|select|confirmation|confirm|competition-audit" >&2
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

METHOD="liars_bench_prompt_router_v1"
ARTIFACT_ROOT="results/blackbox/${METHOD}"
SWEEP_ROOT="results/blackbox/liars_bench_prompt_sweep_v1"
EVAL_ARTIFACT="results/blackbox/liars_bench_ood_gptoss_v1/eval.jsonl"
PARENT_ADAPTER="results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter"
CONTROL_CONFIG="configs/liars_bench_prompt_control.yaml"
SPECIALIST_CONFIG="configs/liars_bench_prompt_honest_alternative.yaml"
LOG_DIR="logs/lambda/${METHOD}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${PHASE}-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

require_paths() {
  local path
  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      echo "required path is missing: ${path}" >&2
      exit 1
    fi
  done
}

selected_route() {
  python - "$1" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1]))["selected"]
if value is not None:
    print(value)
PY
}

route_kind() {
  case "$1" in
    knowledge_only) echo "knowledge" ;;
    choice_only) echo "choice" ;;
    knowledge_choice_union) echo "union" ;;
    *)
      echo "unknown selected route: $1" >&2
      exit 1
      ;;
  esac
}

case "${PHASE}" in
  development)
    require_paths \
      "${EVAL_ARTIFACT}" \
      "${SWEEP_ROOT}/development/control.jsonl" \
      "${SWEEP_ROOT}/development/honest_alternative.jsonl"
    python experiments/liars_bench_distillation/compose_prompt_router.py \
      --eval-artifact "${EVAL_ARTIFACT}" \
      --control "${SWEEP_ROOT}/development/control.jsonl" \
      --specialist "${SWEEP_ROOT}/development/honest_alternative.jsonl" \
      --output-dir "${ARTIFACT_ROOT}/development" \
      --split development \
      --expected-rows 400 \
      --route "knowledge_only=knowledge" \
      --route "choice_only=choice" \
      --route "knowledge_choice_union=union"
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
      "${ARTIFACT_ROOT}/selection-development.json" \
      "${CONTROL_CONFIG}" \
      "${SPECIALIST_CONFIG}"
    SELECTED="$(selected_route "${ARTIFACT_ROOT}/selection-development.json")"
    if [[ -z "${SELECTED}" ]]; then
      echo "no development route passed; confirmation is intentionally skipped"
      exit 0
    fi
    KIND="$(route_kind "${SELECTED}")"
    python experiments/liars_bench_distillation/evaluate_prompt_router.py \
      --eval-artifact "${EVAL_ARTIFACT}" \
      --adapter-dir "${PARENT_ADAPTER}" \
      --control-config "${CONTROL_CONFIG}" \
      --specialist-config "${SPECIALIST_CONFIG}" \
      --condition-name "${SELECTED}" \
      --route-kind "${KIND}" \
      --output-dir "${ARTIFACT_ROOT}/confirmation" \
      --split confirmation \
      --expected-rows 400
    ;;
  confirm)
    require_paths \
      "${ARTIFACT_ROOT}/selection-development.json" \
      "${ARTIFACT_ROOT}/confirmation/result.json"
    SELECTED="$(selected_route "${ARTIFACT_ROOT}/selection-development.json")"
    python experiments/liars_bench_distillation/select_prompt_sweep.py \
      --result "${ARTIFACT_ROOT}/confirmation/result.json" \
      --generation-dir "${ARTIFACT_ROOT}/confirmation" \
      --minimum-macro-gain 0.02 \
      --minimum-category-delta -0.02 \
      --minimum-category-source-model-delta -0.05 \
      --maximum-parse-error-increase 10 \
      --output "${ARTIFACT_ROOT}/selection-confirmation.json"
    CONFIRMED="$(selected_route "${ARTIFACT_ROOT}/selection-confirmation.json")"
    if [[ "${CONFIRMED}" != "${SELECTED}" ]]; then
      echo "development route did not pass confirmation"
    fi
    ;;
  competition-audit)
    require_paths \
      "${ARTIFACT_ROOT}/selection-confirmation.json" \
      "dev_splits/dry.train.yaml" \
      "dev_splits/dry.validation.yaml" \
      "dev_splits/dry.test.yaml"
    SELECTED="$(selected_route "${ARTIFACT_ROOT}/selection-confirmation.json")"
    if [[ -z "${SELECTED}" ]]; then
      echo "no route passed confirmation; competition audit is skipped"
      exit 0
    fi
    python experiments/liars_bench_distillation/audit_prompt_router_competition.py \
      --output "${ARTIFACT_ROOT}/competition-coverage.json"
    ;;
  *)
    echo "unknown phase: ${PHASE}" >&2
    exit 2
    ;;
esac
