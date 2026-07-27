#!/bin/bash
# Score the frozen HP-KR prompt route with continuous 0/1 label margins.

set -euo pipefail

PHASE="${1:-}"
if [[ "${PHASE}" != "development" && "${PHASE}" != "confirmation" ]]; then
  echo "usage: $0 development|confirmation" >&2
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

METHOD="liars_bench_prompt_router_auroc_v1"
ARTIFACT_ROOT="results/blackbox/${METHOD}"
EVAL_ARTIFACT="results/blackbox/liars_bench_ood_gptoss_v1/eval.jsonl"
ADAPTER="results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter"
CONTROL_CONFIG="configs/liars_bench_prompt_control.yaml"
SPECIALIST_CONFIG="configs/liars_bench_prompt_honest_alternative.yaml"
LOG_DIR="logs/lambda/${METHOD}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${PHASE}-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

EXTRA_ARGS=()
if [[ "${PHASE}" == "confirmation" ]]; then
  EXTRA_ARGS+=(--selection "${ARTIFACT_ROOT}/development/selection.json")
fi

python experiments/liars_bench_distillation/evaluate_prompt_router_auroc.py \
  --eval-artifact "${EVAL_ARTIFACT}" \
  --adapter-dir "${ADAPTER}" \
  --control-config "${CONTROL_CONFIG}" \
  --specialist-config "${SPECIALIST_CONFIG}" \
  --route-kind knowledge \
  --output-dir "${ARTIFACT_ROOT}/${PHASE}" \
  --split "${PHASE}" \
  --expected-rows 400 \
  --minimum-macro-gain 0.005 \
  "${EXTRA_ARGS[@]}"
