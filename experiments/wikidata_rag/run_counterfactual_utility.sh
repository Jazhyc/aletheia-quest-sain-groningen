#!/bin/bash
#SBATCH --job-name=aq-wikidata-utility
#SBATCH --time=02:00:00
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
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOG_DIR="logs/slurm/wikidata_rag"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/counterfactual-utility-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

ROOT="${OUTPUT_ROOT:-results/blackbox/wikidata_rag_counterfactual_utility_v1}"
ADAPTER="${ADAPTER_DIR:-results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter}"
LIMIT_ARGS=()
if [[ -n "${LIMIT_ROWS:-}" ]]; then
  LIMIT_ARGS=(--limit-rows "${LIMIT_ROWS}")
fi

python experiments/wikidata_rag/score_counterfactual_utility.py \
  --adapter-dir "${ADAPTER}" \
  --input results/blackbox/wikidata_rag_gptoss_supervision_v1/train_candidates.jsonl \
  --supervision-input results/blackbox/wikidata_rag_gptoss_supervision_v1/train_labels_decisive_full_v1.jsonl \
  --output "${ROOT}/train.jsonl" \
  --report "${ROOT}/train_report.json" \
  --split train \
  --validation-input results/blackbox/wikidata_rag_gptoss_supervision_v1/validation_candidates.jsonl \
  --validation-supervision-input results/blackbox/wikidata_rag_gptoss_supervision_v1/validation_labels_decisive_full_v1.jsonl \
  --validation-output "${ROOT}/validation.jsonl" \
  --validation-report "${ROOT}/validation_report.json" \
  "${LIMIT_ARGS[@]}"
