#!/bin/bash
#SBATCH --job-name=aq-neutral-gptoss-a100x2
#SBATCH --time=01:30:00
#SBATCH --mem=64GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if command -v module >/dev/null 2>&1; then
  module load Python/3.12.3-GCCcore-13.3.0
  module load CUDA/13.2.0
fi

source .venv/bin/activate

export NCCL_IB_DISABLE=1
export NCCL_NET=Socket

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  LOG_DIR="logs/slurm/neutral_contrast_judge"
  LOG_FILE="${LOG_DIR}/pair-judge-a100x2-validation-${SLURM_JOB_ID}.out"
  BOOTSTRAP="logs/slurm/${SLURM_JOB_NAME:-aq-neutral-gptoss-a100x2}-${SLURM_JOB_ID}.bootstrap.out"
  mkdir -p "${LOG_DIR}"
  exec >"${LOG_FILE}" 2>&1
  rm -f "${BOOTSTRAP}"
  echo "job_id=${SLURM_JOB_ID}"
fi

python experiments/neutral_contrast_judge/evaluate_pair_judges.py \
  --tensor-parallel-size 2 \
  "$@"
