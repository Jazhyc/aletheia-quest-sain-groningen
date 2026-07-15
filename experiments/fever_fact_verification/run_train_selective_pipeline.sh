#!/bin/bash
#SBATCH --job-name=aq-fever-train
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/fever_fact_verification
LOG="logs/slurm/fever_fact_verification/train-pipeline-${SLURM_JOB_ID}.out"
exec >"${LOG}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

OUT="results/blackbox/fever_fact_verification_train_v1"
RAG="results/blackbox/privileged_information_distillation_rag"
CLAIMS="${OUT}/atomic_claims/generations.jsonl"
QUESTIONS="${RAG}/wikipedia_varied_train.jsonl"
QUESTION_RETRIEVAL="${OUT}/varied_train_question_retrieval.jsonl"
PAGES="${OUT}/varied_train_top3_full_pages.jsonl"
SINGLES="${OUT}/varied_train_top3_full_retrieval.jsonl"
WINDOWS="${OUT}/varied_train_top3_full_window12_retrieval.jsonl"
SINGLE_RANK="${OUT}/varied_train_top3_full_minilm_top12.jsonl"
WINDOW_RANK="${OUT}/varied_train_top3_full_window12_minilm_top12.jsonl"
INITIAL="${OUT}/varied_train_selective_initial_audit.jsonl"
WINDOW_AUDIT="${OUT}/varied_train_selective_window12_audit.jsonl"
LEXICAL_AUDIT="${OUT}/varied_train_selective_cached_top12_audit.jsonl"
REFERENCE="${RAG}/fever_train_selective_window_lexical_union_v1.jsonl"

python experiments/fever_fact_verification/build_from_question_cache.py \
  --claims "${CLAIMS}" \
  --question-cache "${QUESTIONS}" \
  --output "${QUESTION_RETRIEVAL}" \
  --dataset-name-contains varied-deception

# Seed with already-fetched validation pages. fetch_page_cache safely appends
# only missing successful titles and downstream joins ignore unrelated titles.
if [[ ! -f "${PAGES}" ]]; then
  cp results/blackbox/fever_fact_verification_v1/varied_validation_top3_full_pages.jsonl "${PAGES}"
fi
python experiments/fever_fact_verification/fetch_page_cache.py \
  --retrieval "${QUESTION_RETRIEVAL}" \
  --output "${PAGES}" \
  --max-page-rank 2 \
  --delay 0.5

python experiments/fever_fact_verification/build_from_page_cache.py \
  --retrieval "${QUESTION_RETRIEVAL}" \
  --page-cache "${PAGES}" \
  --output "${SINGLES}" \
  --sentences-per-page 24 \
  --max-page-rank 2 \
  --window-sizes 1
python experiments/fever_fact_verification/build_from_page_cache.py \
  --retrieval "${QUESTION_RETRIEVAL}" \
  --page-cache "${PAGES}" \
  --output "${WINDOWS}" \
  --sentences-per-page 24 \
  --max-page-rank 2 \
  --window-sizes 1,2

python experiments/fever_fact_verification/verify_evidence.py \
  --retrieval "${SINGLES}" \
  --output "${SINGLE_RANK}" \
  --max-candidates 72 \
  --rerank-top-n 12 \
  --top-k 12
python experiments/fever_fact_verification/verify_evidence.py \
  --retrieval "${WINDOWS}" \
  --output "${WINDOW_RANK}" \
  --max-candidates 144 \
  --rerank-top-n 12 \
  --top-k 12

python experiments/fever_fact_verification/audit_selective_hop.py \
  --input "${SINGLE_RANK}" \
  --input-kind verification \
  --max-candidates 5 \
  --output "${INITIAL}"
python experiments/fever_fact_verification/audit_selective_hop.py \
  --input "${WINDOW_RANK}" \
  --input-kind verification \
  --previous-audit "${INITIAL}" \
  --max-candidates 12 \
  --output "${WINDOW_AUDIT}"
python experiments/fever_fact_verification/audit_selective_hop.py \
  --input "${SINGLE_RANK}" \
  --input-kind verification \
  --previous-audit "${INITIAL}" \
  --max-candidates 12 \
  --output "${LEXICAL_AUDIT}"

python experiments/fever_fact_verification/build_selective_reference.py \
  --initial-audit "${INITIAL}" \
  --second-audit "${WINDOW_AUDIT}" \
  --second-audit "${LEXICAL_AUDIT}" \
  --row-universe "${QUESTIONS}" \
  --output "${REFERENCE}"
