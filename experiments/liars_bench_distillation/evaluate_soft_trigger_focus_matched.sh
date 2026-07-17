#!/bin/bash
#SBATCH --job-name=aq-soft-focus-matched
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm/liars_bench_distillation
exec >"logs/slurm/liars_bench_distillation/soft-focus-matched-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  --adapter-dir results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_pid_liars_soft_trigger_replay_continue_adamw2e5_v1/adapter \
  --split validation \
  --run-name validation_liars_soft_trigger_focus_matched_v1

python experiments/liars_bench_distillation/evaluate_students.py \
  --eval-artifact results/blackbox/liars_bench_pid_aug_v1/eval.jsonl \
  --output-dir results/blackbox/liars_bench_soft_trigger_focus_matched_v1/evaluation \
  --adapter baseline=results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter \
  --adapter candidate=results/blackbox/qwen9b_pid_liars_soft_trigger_replay_continue_adamw2e5_v1/adapter
