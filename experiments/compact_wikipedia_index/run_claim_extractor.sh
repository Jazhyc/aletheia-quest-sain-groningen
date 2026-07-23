#!/bin/bash
#SBATCH --job-name=aq-compact-claim-extractor
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/compact_wikipedia_index"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/claim-extractor-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python experiments/compact_wikipedia_index/generate_claim_queries.py \
  --records results/blackbox/gpt_oss_120b_atomic_claim_prompt_sweep_v2/validation/generations.jsonl \
  --output results/blackbox/compact_wikipedia_train_index_v2/validation_qwen_claims.jsonl
