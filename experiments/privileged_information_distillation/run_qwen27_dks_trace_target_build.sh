#!/bin/bash
#SBATCH --job-name=aq-pid-q27trace-build
#SBATCH --time=00:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

LOG_DIR="logs/slurm/privileged_information_distillation"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/qwen27-dks-trace-build-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python "${WORKTREE_ROOT}/experiments/privileged_information_distillation/build_qwen27_dks_trace_targets.py" \
  --generations results/blackbox/qwen27b_reason_ensemble_dks_member4096_softtrain_v1/train/generations.jsonl \
  --generation-config results/blackbox/qwen27b_reason_ensemble_dks_member4096_softtrain_v1/train/config.yaml \
  --direct-distributions results/blackbox/qwen27b_reason_ensemble_dks3072_logit_soft_teacher_v1/train/generations.jsonl \
  --base-teacher results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl \
  --output-dir results/blackbox/qwen27b_dks_full_trace_soft_teacher_v1
