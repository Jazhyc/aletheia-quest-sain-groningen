#!/bin/bash
#SBATCH --job-name=aq-pid-teacher-sweep
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/privileged_information_distillation
LOG="logs/slurm/privileged_information_distillation/teacher-sweep-${SLURM_JOB_ID}.out"
exec >"${LOG}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 CONFIG_NAME [CONFIG_NAME ...]" >&2
  exit 2
fi

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

for config_name in "$@"; do
  echo "teacher_config=${config_name}"
  python experiments/privileged_information_distillation/generate_teacher_data.py \
    --config-name "${config_name}"
done
