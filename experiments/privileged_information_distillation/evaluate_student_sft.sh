#!/bin/bash
#SBATCH --job-name=aq-pid-eval
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

LOG_DIR="logs/slurm/privileged_information_distillation"
mkdir -p "${LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${LOG_DIR}/validation-${SLURM_JOB_ID}.out" 2>&1
  rm -f "logs/slurm/${SLURM_JOB_NAME:-aq-pid-eval}-${SLURM_JOB_ID}.bootstrap.out"
  echo "job_id=${SLURM_JOB_ID}"
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  --adapter-dir results/blackbox/qwen9b_privileged_gptoss120b_summary_adamwlr1e5_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_privileged_gptoss120b_summary_adamwlr5e5_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_privileged_gptoss120b_summary_adamwlr1e4_v1/adapter \
  "$@"
