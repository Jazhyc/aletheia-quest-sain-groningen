#!/bin/bash
#SBATCH --job-name=aq-rag-gate-eval
#SBATCH --time=00:45:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/wikidata_rag
LOG="logs/slurm/wikidata_rag/cross-encoder-gate-eval-${SLURM_JOB_ID}.out"
exec >"${LOG}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

ROOT=results/blackbox/wikidata_rag_counterfactual_utility_matched_reader_v1
REFIT=results/blackbox/wikidata_rag_cross_encoder_minilm_matched_reader_refit_v1
CACHE="${REFIT}/validation_sweep_cache.jsonl"

python experiments/wikidata_rag/build_cross_encoder_sweep_cache.py \
  --training-input "${ROOT}/train.jsonl" \
  --validation-input "${ROOT}/validation.jsonl" \
  --predictions "${REFIT}/selected_predictions.npz" \
  --output "${CACHE}"

python experiments/wikidata_rag/evaluate_matched_reader.py \
  --adapter-dir results/blackbox/qwen9b_pid_wikidata_matched_reader_v1/adapter \
  --retrieval-cache "${CACHE}" \
  --split validation \
  --run-name validation_cross_encoder_gate_v1
