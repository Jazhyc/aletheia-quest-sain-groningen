#!/bin/bash
#SBATCH --job-name=aq-neutral-gptoss-pair
#SBATCH --time=01:30:00
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

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  LOG_DIR="logs/slurm/neutral_contrast_judge"
  LOG_FILE="${LOG_DIR}/pair-judge-validation-${SLURM_JOB_ID}.out"
  BOOTSTRAP="logs/slurm/${SLURM_JOB_NAME:-aq-neutral-gptoss-pair}-${SLURM_JOB_ID}.bootstrap.out"
  mkdir -p "${LOG_DIR}"
  exec >"${LOG_FILE}" 2>&1
  rm -f "${BOOTSTRAP}"
  echo "job_id=${SLURM_JOB_ID}"
fi

python experiments/neutral_contrast_judge/evaluate_pair_judges.py "$@"
