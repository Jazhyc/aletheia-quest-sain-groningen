#!/bin/bash
#SBATCH --job-name=aq-wikidata-weekly
#SBATCH --time=06:00:00
#SBATCH --mem=12GB
#SBATCH --partition=regular
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

LOG_DIR="logs/slurm/wikidata_rag"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/weekly-index-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

python experiments/wikidata_rag/build_broad_index.py \
  --output-dir results/blackbox/wikidata_rag_weekly_v1 \
  --start 2021-01 \
  --end 2024-12 \
  --day-step 7 \
  --max-cards 100000 \
  --workers 3 \
  --delay-seconds 0.4
