#!/bin/bash
# Run the AUROC-first Phoenix margin sweep on the persistent Lambda H100.

set -euo pipefail

PHASE="${1:-validation}"
if [[ "${PHASE}" != "validation" && "${PHASE}" != "test" ]]; then
  echo "usage: $0 [validation|test]" >&2
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

METHOD="phoenix_v3_auroc_margin_sweep_v1"
ADAPTER_ROOT="results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1"
ADAPTER="${ADAPTER_ROOT}/adapter"
RUN_NAME="${PHASE}_${METHOD}"
OUTPUT_DIR="${ADAPTER_ROOT}/${RUN_NAME}"
LOG_DIR="logs/lambda/${METHOD}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${PHASE}-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

for path in \
  "${ADAPTER}/adapter_config.json" \
  "${ADAPTER_ROOT}/config.yaml" \
  "dev_splits/dry.${PHASE}.yaml"; do
  if [[ ! -e "${path}" ]]; then
    echo "required path is missing: ${path}" >&2
    exit 1
  fi
done

ARGS=(
  --adapter-dir "${ADAPTER}"
  --split "${PHASE}"
  --run-name "${RUN_NAME}"
  --continuous-margins
)
if [[ "${PHASE}" == "test" ]]; then
  ARGS+=(
    --continuous-margin-condition direct
    --max-new-tokens 1
  )
fi

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  "${ARGS[@]}"

python experiments/privileged_information_distillation/analyze_continuous_margins.py \
  "${OUTPUT_DIR}/generations.jsonl" \
  | tee "${OUTPUT_DIR}/margin_analysis.json"
