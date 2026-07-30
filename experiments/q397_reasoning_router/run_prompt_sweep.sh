#!/bin/bash
#SBATCH --job-name=aq-q397-reason-prompts
#SBATCH --time=01:00:00
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

LOG_DIR="logs/slurm/q397_reasoning_router"
mkdir -p "${LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${LOG_DIR}/validation-${SLURM_JOB_ID}.out" 2>&1
  rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"
  echo "job_id=${SLURM_JOB_ID}"
fi

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  --adapter-dir results/blackbox/qwen9b_qwen397_tvg_soft_r16_lr5e5_ep2_v1/adapter \
  --split validation \
  --run-name validation_reasoning_prompt_sweep_v1 \
  --max-new-tokens 192 \
  --max-model-len 4096 \
  --continuous-margins \
  --continuous-margin-condition direct \
  --continuous-margin-condition reasoning \
  --prompt-condition binary=experiments/q397_reasoning_router/prompts/binary.yaml \
  --prompt-condition summary_baseline=experiments/q397_reasoning_router/prompts/summary_baseline.yaml \
  --prompt-condition claim_check=experiments/q397_reasoning_router/prompts/claim_check.yaml \
  --prompt-condition balanced_audit=experiments/q397_reasoning_router/prompts/balanced_audit.yaml
