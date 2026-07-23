#!/bin/bash
#SBATCH --job-name=aq-hetero-test
#SBATCH --time=01:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/heterogeneous_adapter_ensemble"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/evaluate-test-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python experiments/heterogeneous_adapter_ensemble/evaluate_members.py \
  --member deception=results/blackbox/qwen9b_pid_specialist_material_rank1_v1/adapter \
  --member incorrectness=results/blackbox/qwen9b_heterogeneous_incorrectness_rank1_v1/adapter \
  --member resolved_intent=results/blackbox/qwen9b_heterogeneous_resolved_intent_rank1_v1/adapter \
  --requires-reasoning resolved_intent \
  --max-new-tokens 512 \
  --max-model-len 4096 \
  --split test \
  --run-name test_heterogeneous_objectives_v1
