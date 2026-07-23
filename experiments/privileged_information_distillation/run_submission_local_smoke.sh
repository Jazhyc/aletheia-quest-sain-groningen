#!/bin/bash
#SBATCH --job-name=aq-pid-submission-smoke
#SBATCH --time=00:30:00
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
exec >"${LOG_DIR}/submission-smoke-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTHONUNBUFFERED=1
export DATASET_NAME="${DATASET_NAME:-aletheias-quest/dev-varied-deception-Qwen3.5-27B-None}"
export ALETHEIA_LIMIT="${ALETHEIA_LIMIT:-32}"
export PHOENIX_NNSIGHT_LOCAL=1

mapfile -t NOTEBOOKS < <(find submission -maxdepth 1 -type f -name '*.ipynb' -print | sort)
if (( ${#NOTEBOOKS[@]} != 1 )); then
  echo "submission/ must contain exactly one notebook; found ${#NOTEBOOKS[@]}" >&2
  exit 1
fi

TEMP_NOTEBOOK=".phoenix-wright-smoke-${SLURM_JOB_ID}.ipynb"
EXECUTED_NOTEBOOK=".phoenix-wright-smoke-${SLURM_JOB_ID}.executed.ipynb"
cleanup() {
  rm -f "${TEMP_NOTEBOOK}" "${EXECUTED_NOTEBOOK}"
}
trap cleanup EXIT TERM INT
cp "${NOTEBOOKS[0]}" "${TEMP_NOTEBOOK}"
jupyter execute --timeout=-1 --output="${EXECUTED_NOTEBOOK%.ipynb}" "${TEMP_NOTEBOOK}"
