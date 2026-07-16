#!/bin/bash
#SBATCH --job-name=aq-action-prompt
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/liars_bench_distillation
exec >"logs/slurm/liars_bench_distillation/action-prompt-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

python experiments/liars_bench_distillation/evaluate_action_prompt.py \
  --eval-artifact results/blackbox/liars_bench_pid_aug_v1/eval.jsonl \
  --output-dir results/blackbox/liars_bench_pid_aug_v1/action_prompt_4500 \
  --adapter baseline=results/blackbox/qwen9b_pid_varied_datafrac10_adamw5e5_v1/adapter \
  --adapter observable2=results/blackbox/qwen9b_pid_varied2_liars_observable_aug_adamw5e5_v1/adapter \
  --adapter observable=results/blackbox/qwen9b_pid_varied10_liars_observable_aug_adamw5e5_v1/adapter \
  --adapter broad=results/blackbox/qwen9b_pid_varied10_liars_broad_aug_adamw5e5_v1/adapter \
  --adapter broad_continue=results/blackbox/qwen9b_pid_liars_broad_replay_continue_adamw2e5_v1/adapter
