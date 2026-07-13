#!/bin/bash
#SBATCH --job-name=wikidata-expanded-v2
#SBATCH --partition=regular
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/wikidata_rag/expanded-%j.out

set -euo pipefail

cd /scratch/s4626451/Aletheias-Quest-Competition
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate
python experiments/wikidata_rag/build_expanded_index.py \
  --seed-cards results/blackbox/wikidata_rag_daily_v1/cache/cards.jsonl \
  --output-dir results/blackbox/wikidata_rag_expanded_v2 \
  --workers 3 \
  --delay-seconds 0.1
