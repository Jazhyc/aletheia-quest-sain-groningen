#!/bin/bash
#SBATCH --job-name=aq-action-recipient
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/liars_bench_distillation
exec >"logs/slurm/liars_bench_distillation/action-recipient-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

python experiments/liars_bench_distillation/evaluate_action_recipient_clarification.py \
  --adapter results/blackbox/qwen9b_pid_varied_datafrac10_adamw5e5_v1/adapter \
  --baseline results/blackbox/liars_bench_pid_aug_v1/action_prompt_full_confirmation/action.jsonl \
  --output-dir results/blackbox/liars_bench_pid_aug_v1/action_recipient_full
