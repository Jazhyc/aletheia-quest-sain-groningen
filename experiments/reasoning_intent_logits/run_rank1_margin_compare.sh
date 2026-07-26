#!/bin/bash
#SBATCH --job-name=aq-intent-r1-margin
#SBATCH --time=00:45:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if command -v module >/dev/null 2>&1; then
  module load Python/3.12.3-GCCcore-13.3.0
  module load CUDA/13.2.0
fi

source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

SPLIT="${1:-validation}"
if [[ "${SPLIT}" != "validation" && "${SPLIT}" != "test" ]]; then
  echo "usage: sbatch $0 [validation|test]" >&2
  exit 2
fi

METHOD_LOG_DIR="logs/slurm/reasoning_intent_logits"
METHOD_LOG_FILE="${METHOD_LOG_DIR}/${SPLIT}-rank1-margin-${SLURM_JOB_ID}.out"
BOOTSTRAP_LOG_FILE="logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"
mkdir -p "${METHOD_LOG_DIR}"
echo "Redirecting Slurm job output to ${METHOD_LOG_FILE}"
exec >"${METHOD_LOG_FILE}" 2>&1
rm -f "${BOOTSTRAP_LOG_FILE}"

ADAPTER_DIR="results/blackbox/qwen9b_heterogeneous_resolved_intent_rank1_v1/adapter"
if [[ "${SPLIT}" == "validation" ]]; then
  RUN_NAME="validation_direct_intent_margins_v1"
  MARGIN_CONDITIONS=(direct empty)
else
  RUN_NAME="test_empty_intent_margin_v1"
  MARGIN_CONDITIONS=(empty)
fi

MARGIN_ARGS=()
for condition in "${MARGIN_CONDITIONS[@]}"; do
  MARGIN_ARGS+=(--continuous-margin-condition "${condition}")
done

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  --adapter-dir "${ADAPTER_DIR}" \
  --split "${SPLIT}" \
  --run-name "${RUN_NAME}" \
  --max-new-tokens 1 \
  --continuous-margins \
  "${MARGIN_ARGS[@]}"
