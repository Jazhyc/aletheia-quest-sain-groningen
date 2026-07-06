#!/bin/bash
#SBATCH --job-name=aq-qwen-grpo-eval
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

METHOD="qwen_grpo_lora_r16_reasonfield_v1"
PY_ARGS=()
for arg in "$@"; do
  if [[ "${arg}" == method=* ]]; then
    METHOD="${arg#method=}"
  else
    PY_ARGS+=("${arg}")
  fi
done
METHOD_LOG_DIR="logs/slurm/${METHOD}"
mkdir -p "${METHOD_LOG_DIR}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  METHOD_LOG_FILE="${METHOD_LOG_DIR}/eval-${SLURM_JOB_ID}.out"
  BOOTSTRAP_LOG_FILE="logs/slurm/${SLURM_JOB_NAME:-aq-qwen-grpo-eval}-${SLURM_JOB_ID}.bootstrap.out"
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
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-/scratch/${USER}}/.cache/uv}"

python experiments/qwen_grpo_lora/evaluate_qwen_grpo_lora.py "${PY_ARGS[@]}"
