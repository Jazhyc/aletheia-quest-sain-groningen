#!/bin/bash
#SBATCH --job-name=aq-pid-r1-specialist-eval
#SBATCH --time=01:15:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/pid_specialist_ensemble"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/evaluate-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

ADAPTER_ARGS=(
  --adapter-dir results/blackbox/qwen9b_pid_specialist_material_rank1_v1/adapter
  --adapter-dir results/blackbox/qwen9b_pid_specialist_polarity_rank1_v1/adapter
  --adapter-dir results/blackbox/qwen9b_pid_specialist_hierarchy_rank1_v1/adapter
  --max-new-tokens 512
  --max-model-len 4096
)

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  "${ADAPTER_ARGS[@]}" --split train --run-name train_meta_features_v1
python experiments/privileged_information_distillation/evaluate_student_sft.py \
  "${ADAPTER_ARGS[@]}" --split validation --run-name validation_specialist_ensemble_v1
