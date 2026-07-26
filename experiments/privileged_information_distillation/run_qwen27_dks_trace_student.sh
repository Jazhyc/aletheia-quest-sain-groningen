#!/bin/bash
#SBATCH --job-name=aq-pid-q27trace-student
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate

LOG_DIR="logs/slurm/privileged_information_distillation"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/qwen27-dks-trace-student-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

WORKTREE_ROOT="${QWEN27_DKS_WORKTREE:?missing QWEN27_DKS_WORKTREE}"
python "${WORKTREE_ROOT}/experiments/privileged_information_distillation/train_student_sft.py" \
  --config-name pid_qwen27_dks_full_trace_soft90_v1 \
  "$@"
