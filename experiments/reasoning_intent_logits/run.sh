#!/bin/bash
#SBATCH --job-name=aq-reason-intent
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

SPLIT="validation"
for argument in "$@"; do
  if [[ "${argument}" == split=* ]]; then
    SPLIT="${argument#split=}"
  fi
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  METHOD_LOG_DIR="logs/slurm/reasoning_intent_logits"
  METHOD_LOG_FILE="${METHOD_LOG_DIR}/${SPLIT}-${SLURM_JOB_ID}.out"
  BOOTSTRAP_LOG_FILE="logs/slurm/${SLURM_JOB_NAME:-aq-reason-intent}-${SLURM_JOB_ID}.bootstrap.out"
  mkdir -p "${METHOD_LOG_DIR}"
  echo "Redirecting Slurm job output to ${METHOD_LOG_FILE}"
  exec >"${METHOD_LOG_FILE}" 2>&1
  rm -f "${BOOTSTRAP_LOG_FILE}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "split=${SPLIT}"
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

python experiments/blackbox/run_judge.py \
  --config-name judge_ensemble/blackbox_reasoning_intent_logits_v1 \
  "$@"
