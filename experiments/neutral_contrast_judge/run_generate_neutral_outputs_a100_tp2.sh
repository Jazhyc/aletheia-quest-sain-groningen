#!/bin/bash
#SBATCH --job-name=aq-neutral-out-a100x2
#SBATCH --time=01:00:00
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
  LOG_FILE="${LOG_DIR}/neutral-outputs-validation-${SLURM_JOB_ID}.out"
  BOOTSTRAP="logs/slurm/${SLURM_JOB_NAME:-aq-neutral-out-a100x2}-${SLURM_JOB_ID}.bootstrap.out"
  mkdir -p "${LOG_DIR}"
  exec >"${LOG_FILE}" 2>&1
  rm -f "${BOOTSTRAP}"
  echo "job_id=${SLURM_JOB_ID}"
fi

python experiments/neutral_contrast_judge/generate_neutral_outputs.py \
  --tensor-parallel-size 2 \
  --max-num-seqs 128 \
  "$@"
