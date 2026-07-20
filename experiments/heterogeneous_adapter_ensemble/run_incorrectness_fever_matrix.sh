#!/bin/bash
#SBATCH --job-name=aq-knowledge-rag
#SBATCH --time=00:45:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/heterogeneous_adapter_ensemble"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/knowledge-rag-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

python experiments/wikidata_rag/evaluate_matched_reader.py \
  --adapter-dir results/blackbox/qwen9b_heterogeneous_incorrectness_rank1_v1/adapter \
  --retrieval-cache results/blackbox/fever_fact_verification_v1/varied_validation_selective_expanded_window_union_reference_v1.jsonl \
  --condition empty \
  --condition real \
  --condition shuffled \
  --run-name validation_fever_evidence_v1 \
  --max-new-tokens 512 \
  --max-model-len 4096
