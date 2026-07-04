#!/bin/bash
#SBATCH --job-name=aq-logit-sweep
#SBATCH --time=04:00:00
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

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  METHOD_LOG_DIR="logs/slurm/logit_prompt_sweep"
  METHOD_LOG_FILE="${METHOD_LOG_DIR}/validation-${SLURM_JOB_ID}.out"
  BOOTSTRAP_LOG_FILE="logs/slurm/${SLURM_JOB_NAME:-aq-logit-sweep}-${SLURM_JOB_ID}.bootstrap.out"
  mkdir -p "${METHOD_LOG_DIR}"
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

run_candidate() {
  local config_name="$1"
  shift
  echo
  echo "===== $(date --iso-8601=seconds) ${config_name} $* ====="
  python experiments/blackbox/run_judge.py --config-name "${config_name}" "$@"
}

run_candidate judge_ensemble/blackbox_reasoning_ensemble_dk3072_logit_outcome_v1 \
  split=validation

run_candidate judge_ensemble/blackbox_reasoning_ensemble_dk3072_logit_outcome_v1 \
  method=qwen_reason_ensemble_dk3072_logit_outcome_headtail_v1 \
  split=validation \
  +judge.context_truncation=head_tail

run_candidate judge_ensemble/blackbox_reasoning_ensemble_yesno3072_logit_v1 \
  split=validation

run_candidate judge_ensemble/blackbox_reasoning_ensemble_yesno3072_logit_v1 \
  method=qwen_reason_ensemble_yesno3072_logit_headtail_v1 \
  split=validation \
  +judge.context_truncation=head_tail
