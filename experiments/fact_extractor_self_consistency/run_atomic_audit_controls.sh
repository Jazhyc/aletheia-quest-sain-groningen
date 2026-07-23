#!/bin/bash
#SBATCH --job-name=aq-atomic-controls
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

METHOD="fact_extractor_self_consistency"
LOG_DIR="logs/slurm/${METHOD}"
mkdir -p "${LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${LOG_DIR}/atomic-controls-validation-${SLURM_JOB_ID}.out" 2>&1
  rm -f "logs/slurm/${SLURM_JOB_NAME:-aq-atomic-controls}-${SLURM_JOB_ID}.bootstrap.out"
  echo "job_id=${SLURM_JOB_ID}"
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

python experiments/fact_extractor_self_consistency/run_atomic_audit_controls.py \
  --model openai/gpt-oss-120b \
  --output-name gpt_oss_120b_atomic_audit_controls_validation_v1

python experiments/fact_extractor_self_consistency/run_atomic_audit_controls.py \
  --model Qwen/Qwen3.5-9B \
  --output-name qwen9b_atomic_audit_controls_validation_v1
