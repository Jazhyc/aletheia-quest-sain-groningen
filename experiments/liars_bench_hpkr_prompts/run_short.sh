#!/bin/bash
#SBATCH --job-name=aq-hpkr-short
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.2.0
source .venv/bin/activate
set -a
source .env
set +a

LOG_DIR="logs/slurm/liars_bench_hpkr_prompts"
mkdir -p "${LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${LOG_DIR}/short-${SLURM_JOB_ID}.out" 2>&1
  rm -f "logs/slurm/${SLURM_JOB_NAME:-aq-hpkr-short}-${SLURM_JOB_ID}.bootstrap.out"
  echo "job_id=${SLURM_JOB_ID}"
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

python experiments/liars_bench_hpkr_prompts/evaluate_short.py --model Qwen/Qwen3.5-9B --condition-name base_qwen_short
python experiments/liars_bench_hpkr_prompts/evaluate_short.py --model trohrbaugh/Qwen3.5-9B-heretic-v2 --condition-name heretic_v2_short
