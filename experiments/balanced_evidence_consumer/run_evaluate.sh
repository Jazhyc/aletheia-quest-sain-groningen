#!/bin/bash
#SBATCH --job-name=aq-balanced-eval
#SBATCH --time=00:45:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/balanced_evidence_consumer
LOG="logs/slurm/balanced_evidence_consumer/evaluate-${SLURM_JOB_ID}.out"
exec >"${LOG}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CACHE="${1:-results/blackbox/compact_wikipedia_train_index_v2/validation_learned_cache.jsonl}"
RUN_NAME="${2:-validation_compact_wikipedia_balanced_consumer_rank1_v1}"

python experiments/wikidata_rag/evaluate_matched_reader.py \
  --adapter-dir results/blackbox/qwen9b_fever_balanced_evidence_consumer_rank1_v1/adapter \
  --prompt-config configs/privileged_information_distillation_fever_balanced_rank1.yaml \
  --retrieval-cache "${CACHE}" \
  --condition empty \
  --condition real \
  --condition shuffled \
  --run-name "${RUN_NAME}" \
  --max-new-tokens 512 \
  --max-model-len 4096
