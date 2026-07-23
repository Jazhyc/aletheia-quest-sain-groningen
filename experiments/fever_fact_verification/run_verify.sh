#!/bin/bash
#SBATCH --job-name=fever-verify
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs/slurm/fever_fact_verification
LOG="logs/slurm/fever_fact_verification/validation-${SLURM_JOB_ID}.out"
exec >"$LOG" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

export HF_HOME="${HF_HOME:-/scratch/$USER/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false

.venv/bin/python experiments/fever_fact_verification/verify_evidence.py "$@"
