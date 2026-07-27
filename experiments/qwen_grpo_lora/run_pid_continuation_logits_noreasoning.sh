#!/bin/bash
#SBATCH --job-name=aq-pid-grpo-logits
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail

CODE_ROOT="${AQ_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
COMMON_GIT_DIR="$(git -C "${CODE_ROOT}" rev-parse --git-common-dir)"
SHARED_ROOT="$(cd -- "$(dirname -- "${COMMON_GIT_DIR}")" && pwd)"

METHOD="pid_grpo_continuation_logits_noreasoning"
METHOD_LOG_DIR="${SHARED_ROOT}/logs/slurm/${METHOD}"
METHOD_LOG_FILE="${METHOD_LOG_DIR}/validation-${SLURM_JOB_ID}.out"
BOOTSTRAP_LOG_FILE="${CODE_ROOT}/logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"
mkdir -p "${METHOD_LOG_DIR}"
exec >"${METHOD_LOG_FILE}" 2>&1
rm -f "${BOOTSTRAP_LOG_FILE}"

echo "job_id=${SLURM_JOB_ID}"
echo "code_root=${CODE_ROOT}"
echo "shared_root=${SHARED_ROOT}"

if command -v module >/dev/null 2>&1; then
  module load Python/3.12.3-GCCcore-13.3.0
  module load CUDA/13.2.0
fi

source "${SHARED_ROOT}/.venv/bin/activate"

if [[ -f "${SHARED_ROOT}/.env" ]]; then
  set -a
  source "${SHARED_ROOT}/.env"
  set +a
fi

export TOKENIZERS_PARALLELISM=false
cd "${SHARED_ROOT}"

run_epoch() {
  local epoch="$1"
  local method="qwen9b_pid_varied_grpo_ep${epoch}_v1"
  echo
  echo "===== $(date --iso-8601=seconds) ${method} ====="
  python "${CODE_ROOT}/experiments/qwen_grpo_lora/evaluate_qwen_grpo_lora_logits.py" \
    --adapter-dir "${SHARED_ROOT}/results/blackbox/${method}/adapter" \
    --split validation \
    --splits-dir "${SHARED_ROOT}/dev_splits" \
    --output-dir \
      "${SHARED_ROOT}/results/blackbox/${method}/validation_logits_empty_reasoning_plain_noreasoning" \
    --batch-size 4 \
    --prefix-variant empty_reasoning \
    --label-style plain \
    --exclude-reasoning
}

run_epoch 1
run_epoch 2
