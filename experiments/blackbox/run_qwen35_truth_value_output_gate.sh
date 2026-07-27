#!/bin/bash
#SBATCH --job-name=aq-qwen35-tvg-r7
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

METHOD="qwen35_truth_value_output_gate"
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

cd "${SHARED_ROOT}"

python "${CODE_ROOT}/experiments/blackbox/run_judge.py" \
  --config-name \
  single_judges/blackbox_reasoning_nothink_truth_value_rating7_logit_qwen35_27b_v1 \
  split=validation \
  "splits_dir=${SHARED_ROOT}/dev_splits" \
  "output_dir=${SHARED_ROOT}/results/blackbox" \
  "leaderboard=${SHARED_ROOT}/results/blackbox/leaderboard.md"
