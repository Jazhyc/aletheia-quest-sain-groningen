#!/bin/bash
#SBATCH --job-name=aq-gemma-route-confirm
#SBATCH --time=00:45:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm/liars_bench_distillation
exec >"logs/slurm/liars_bench_distillation/gemma-route-confirm-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

python experiments/liars_bench_distillation/evaluate_soft_trigger_gemma_confirmation.py \
  --eval-artifact results/blackbox/liars_bench_soft_trigger_gemma_confirmation_v1/eval.jsonl \
  --output-dir results/blackbox/liars_bench_soft_trigger_gemma_confirmation_v1/evaluation \
  --adapter baseline=results/blackbox/qwen9b_pid_varied_datafrac10_adamw5e5_v1/adapter \
  --adapter candidate=results/blackbox/qwen9b_pid_liars_soft_trigger_replay_continue_adamw2e5_v1/adapter
