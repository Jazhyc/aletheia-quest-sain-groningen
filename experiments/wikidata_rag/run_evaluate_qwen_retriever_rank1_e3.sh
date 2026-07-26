#!/bin/bash
#SBATCH --job-name=aq-qwen-ret-e3ev
#SBATCH --time=00:35:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/wikidata_rag
LOG="logs/slurm/wikidata_rag/qwen-retriever-e3-eval-${SLURM_JOB_ID}.out"
exec >"${LOG}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

python -m experiments.wikidata_rag.evaluate_qwen_retriever_rank1 \
  --adapter-dir results/blackbox/wikidata_qwen_retriever_rank1_e3_v1/adapter \
  --train-labels results/blackbox/wikidata_rag_gptoss_supervision_v1/train_labels_decisive_full_v1.jsonl \
  --validation-labels results/blackbox/wikidata_rag_gptoss_supervision_v1/validation_labels_decisive_full_v1.jsonl \
  --validation-planner results/blackbox/wikidata_rag_qwen_planner_v1/validation_full_v1.jsonl \
  --output-dir results/blackbox/wikidata_qwen_retriever_rank1_e3_v1/evaluation \
  --adapter-only
