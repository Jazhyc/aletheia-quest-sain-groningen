#!/bin/bash
#SBATCH --job-name=aq-pid-student
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/privileged_information_distillation
LOG="logs/slurm/privileged_information_distillation/student-${SLURM_JOB_ID}.out"
exec >"${LOG}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

python experiments/privileged_information_distillation/train_student_sft.py "$@"

if [[ -n "${QWEN35_CANONICALIZE_ADAPTER:-}" ]]; then
  ADAPTER_PATH="$(realpath -m "${QWEN35_CANONICALIZE_ADAPTER}")"
  case "${ADAPTER_PATH}" in
    "${PWD}/results/blackbox/"*) ;;
    *)
      echo "refusing to canonicalize adapter outside results/blackbox: ${ADAPTER_PATH}" >&2
      exit 1
      ;;
  esac
  if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" ]]; then
    echo "missing trained adapter weights: ${ADAPTER_PATH}" >&2
    exit 1
  fi
  WORK_DIR="${QWEN35_CANONICALIZATION_WORK_DIR:-${ADAPTER_PATH%/adapter}/peft_path_migration}"
  python experiments/heterogeneous_adapter_ensemble/migrate_qwen35_peft_paths.py \
    --local-only \
    --local-dir "${ADAPTER_PATH}" \
    --work-dir "${WORK_DIR}"
fi
