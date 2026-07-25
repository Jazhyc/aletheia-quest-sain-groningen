#!/bin/bash
# Train and validate matched AUROC-loss ablations on the persistent Lambda H100.

set -euo pipefail

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

CAMPAIGN_METHOD="qwen9b_pid_auroc_loss_ablation_v1"
LOG_DIR="logs/lambda/${CAMPAIGN_METHOD}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/validation-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

TEACHER_CACHE="results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl"
if [[ ! -f "${TEACHER_CACHE}" ]]; then
  echo "required teacher cache is missing: ${TEACHER_CACHE}" >&2
  exit 1
fi

METHODS=(
  qwen9b_pid_auroc_reasoning_paired_v1
  qwen9b_pid_auroc_directce_v1
  qwen9b_pid_auroc_rank01_v1
  qwen9b_pid_auroc_rank03_v1
)
DIRECT_WEIGHTS=(0.0 1.0 1.0 1.0)
PAIRWISE_WEIGHTS=(0.0 0.0 0.1 0.3)

for index in "${!METHODS[@]}"; do
  method="${METHODS[$index]}"
  adapter="results/blackbox/${method}/adapter"
  if [[ -f "${adapter}/adapter_config.json" ]]; then
    echo "adapter already complete; skipping training: ${method}"
    continue
  fi
  python experiments/privileged_information_distillation/train_student_sft.py \
    --config-name privileged_information_distillation_auroc_loss \
    "method=${method}" \
    "output_dir=results/blackbox/${method}" \
    "student.training.direct_loss_weight=${DIRECT_WEIGHTS[$index]}" \
    "student.training.pairwise_loss_weight=${PAIRWISE_WEIGHTS[$index]}"
done

EVAL_ARGS=(
  --split validation
  --run-name validation_direct_margin
  --max-new-tokens 1
  --continuous-margins
  --continuous-margin-condition direct
)
for method in "${METHODS[@]}"; do
  EVAL_ARGS+=(--adapter-dir "results/blackbox/${method}/adapter")
done

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  "${EVAL_ARGS[@]}"

for method in "${METHODS[@]}"; do
  output="results/blackbox/${method}/validation_direct_margin"
  python experiments/privileged_information_distillation/analyze_continuous_margins.py \
    "${output}/generations.jsonl" \
    | tee "${output}/margin_analysis.json"
done
