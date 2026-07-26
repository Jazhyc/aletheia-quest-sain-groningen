#!/bin/bash
#SBATCH --job-name=aq-phoenix-oracle
#SBATCH --time=00:45:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/wikidata_rag
LOG="logs/slurm/wikidata_rag/phoenix-candidate-oracle-${SLURM_JOB_ID}.out"
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

ROOT="results/blackbox/wikidata_phoenix_candidate_oracle_v1"
ADAPTER="results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter"
BASELINE="results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/validation_phoenix_v3_auroc_margin_sweep_v1/generations.jsonl"

mkdir -p "${ROOT}"
python -m experiments.wikidata_rag.score_counterfactual_utility \
  --adapter-dir "${ADAPTER}" \
  --input results/blackbox/wikidata_rag_gptoss_supervision_v1/validation_candidates.jsonl \
  --supervision-input results/blackbox/wikidata_rag_gptoss_supervision_v1/validation_labels_decisive_full_v1.jsonl \
  --output "${ROOT}/validation_direct_scores.jsonl" \
  --report "${ROOT}/validation_direct_scores_report.json" \
  --split validation \
  --prefix-mode direct \
  --batch-size 128 \
  --max-model-len 4608

python -m experiments.wikidata_rag.analyze_phoenix_candidate_oracle \
  --candidate-scores "${ROOT}/validation_direct_scores.jsonl" \
  --baseline-generations "${BASELINE}" \
  --output "${ROOT}/result.json"
