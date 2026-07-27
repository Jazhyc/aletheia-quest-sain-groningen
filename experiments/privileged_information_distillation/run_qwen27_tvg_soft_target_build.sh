#!/bin/bash
#SBATCH --job-name=aq-pid-q27tvg-build
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

LOG_DIR="logs/slurm/privileged_information_distillation"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/qwen27-tvg-soft-build-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

WORKTREE_ROOT="${QWEN27_TVG_WORKTREE:?missing QWEN27_TVG_WORKTREE}"
INPUT="results/blackbox/qwen35_27b_nothink_truth_value_binary_logit_v1/train/generations.jsonl"
OUTPUT="results/blackbox/qwen35_27b_nothink_truth_value_binary_logit_v1/train/soft_targets.jsonl"

python "${WORKTREE_ROOT}/experiments/privileged_information_distillation/build_soft_teacher_cache.py" \
  "${INPUT}" \
  "${OUTPUT}" \
  --kind binary_identity \
  --expected-rows 6573
