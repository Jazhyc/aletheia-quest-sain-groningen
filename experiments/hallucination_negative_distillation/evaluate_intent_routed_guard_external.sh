#!/bin/bash
#SBATCH --job-name=aq-hallucination-intent-route
#SBATCH --time=00:45:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/hallucination_negative_distillation
exec >"logs/slurm/hallucination_negative_distillation/intent-route-external-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

python experiments/hallucination_negative_distillation/evaluate_intent_routed_guard_external.py \
  --eval-artifact results/blackbox/liars_bench_pid_aug_v1/eval.jsonl \
  --adapter results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter \
  --baseline-config configs/privileged_information_distillation_reasoning_baseline4000_prompt.yaml \
  --guard-config configs/privileged_information_distillation_reasoning_error_guard4000.yaml \
  --output-dir results/blackbox/hallucination_negative_distillation_v1/intent_routed_guard/external_preservation
