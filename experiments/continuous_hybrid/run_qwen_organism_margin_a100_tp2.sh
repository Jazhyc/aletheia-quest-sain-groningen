#!/bin/bash
#SBATCH --job-name=aq-cont-org-a100x2
#SBATCH --time=01:00:00
# TP2 vLLM plus repeated rank-128 LoRA loads peaked above 33 GB host RAM.
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

# The cluster NCCL network plugin can segfault during same-node TP startup.
# This job needs only local GPU communication, so use NCCL's socket transport.
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  METHOD_LOG_DIR="logs/slurm/continuous_hybrid"
  METHOD_LOG_FILE="${METHOD_LOG_DIR}/validation-a100x2-${SLURM_JOB_ID}.out"
  BOOTSTRAP_LOG_FILE="logs/slurm/${SLURM_JOB_NAME:-aq-cont-org-a100x2}-${SLURM_JOB_ID}.bootstrap.out"
  mkdir -p "${METHOD_LOG_DIR}"
  echo "Redirecting Slurm job output to ${METHOD_LOG_FILE}"
  exec >"${METHOD_LOG_FILE}" 2>&1
  rm -f "${BOOTSTRAP_LOG_FILE}"
  echo "job_id=${SLURM_JOB_ID}"
fi

python experiments/continuous_hybrid/run_qwen_organism_margin.py \
  --tensor-parallel-size 2 \
  --max-num-seqs 128 \
  "$@"
