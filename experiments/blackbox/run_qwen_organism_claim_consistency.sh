#!/bin/bash
#SBATCH --job-name=aq-qwen-claimtf
#SBATCH --time=02:00:00
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

METHOD="qwen27b_organism_atomic_true_false_v1"
LOG_DIR="logs/slurm/${METHOD}"
mkdir -p "${LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${LOG_DIR}/test-${SLURM_JOB_ID}.out" 2>&1
  rm -f "logs/slurm/${SLURM_JOB_NAME:-aq-qwen-claimtf}-${SLURM_JOB_ID}.bootstrap.out"
  echo "job_id=${SLURM_JOB_ID}"
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

python experiments/blackbox/audit_qwen_organism_claim_consistency.py
