#!/bin/bash
#SBATCH --job-name=wikidata-expanded-eval
#SBATCH --partition=regular
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/slurm/wikidata_rag/expanded-eval-%j.out

set -euo pipefail

cd /scratch/s4626451/Aletheias-Quest-Competition
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONPATH=.

OUT=results/blackbox/wikidata_rag_expanded_v2
OLD=results/blackbox/wikidata_rag_daily_v1

python experiments/wikidata_rag/build_claim_gated_cache.py \
  --database "$OUT/wikidata.sqlite" \
  --input "$OLD/train_retrieval_raw.jsonl" \
  --output "$OUT/train_claim_gated_programmatic_v1.jsonl"
python experiments/wikidata_rag/evaluate_claim_gate.py \
  --teacher-cache results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl \
  --broad-cache "$OLD/train_retrieval_raw.jsonl" \
  --gated-cache "$OUT/train_claim_gated_programmatic_v1.jsonl" \
  --output "$OUT/train_claim_gate_programmatic_v1_report.json"
python experiments/wikidata_rag/compare_claim_gate_caches.py \
  --expanded "$OUT/train_claim_gated_programmatic_v1.jsonl" \
  --baseline "$OUT/train_claim_gated.jsonl" \
  --output "$OUT/train_claim_gate_programmatic_v1_comparison.json"

python experiments/wikidata_rag/build_claim_gated_cache.py \
  --database "$OUT/wikidata.sqlite" \
  --input "$OLD/validation_claim_gated_final.jsonl" \
  --output "$OUT/validation_claim_gated_programmatic_v1.jsonl"
python experiments/wikidata_rag/compare_claim_gate_caches.py \
  --expanded "$OUT/validation_claim_gated_programmatic_v1.jsonl" \
  --baseline "$OUT/validation_claim_gated.jsonl" \
  --output "$OUT/validation_claim_gate_programmatic_v1_comparison.json"
