#!/bin/bash
#SBATCH --job-name=aq-minilm-ft
#SBATCH --time=04:00:00
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

METHOD="minilm_finetune_v1"
METHOD_LOG_DIR="logs/slurm/${METHOD}"
mkdir -p "${METHOD_LOG_DIR}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  METHOD_LOG_FILE="${METHOD_LOG_DIR}/train-${SLURM_JOB_ID}.out"
  BOOTSTRAP_LOG_FILE="logs/slurm/${SLURM_JOB_NAME:-aq-minilm-ft}-${SLURM_JOB_ID}.bootstrap.out"
  echo "Redirecting Slurm job output to ${METHOD_LOG_FILE}"
  exec >"${METHOD_LOG_FILE}" 2>&1
  rm -f "${BOOTSTRAP_LOG_FILE}"
  echo "job_id=${SLURM_JOB_ID}"
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export TOKENIZERS_PARALLELISM=false

python experiments/minilm_finetune/run_minilm_finetune.py "$@"
