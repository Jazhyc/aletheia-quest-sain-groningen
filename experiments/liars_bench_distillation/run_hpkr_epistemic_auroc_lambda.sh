#!/bin/bash
# Score the strong base-Qwen knowledge-report specialist with continuous margins.

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

METHOD="liars_bench_hpkr_epistemic_auroc_v1"
ARTIFACT_ROOT="results/blackbox/${METHOD}"
EVAL_ARTIFACT="results/blackbox/liars_bench_ood_gptoss_v1/eval.jsonl"
ADAPTER="results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter"
LOG_DIR="logs/lambda/${METHOD}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${PHASE}-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

EXTRA_ARGS=()
if [[ "${PHASE}" == "confirmation" ]]; then
  EXTRA_ARGS+=(--selection "${ARTIFACT_ROOT}/development/selection.json")
fi

python experiments/liars_bench_distillation/evaluate_hpkr_epistemic_auroc.py \
  --eval-artifact "${EVAL_ARTIFACT}" \
  --phoenix-adapter "${ADAPTER}" \
  --phoenix-config configs/liars_bench_prompt_control.yaml \
  --output-dir "${ARTIFACT_ROOT}/${PHASE}" \
  --split "${PHASE}" \
  --expected-rows 100 \
  --minimum-auroc-gain 0.05 \
  "${EXTRA_ARGS[@]}"
