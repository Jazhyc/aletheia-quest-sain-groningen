#!/bin/bash
#SBATCH --job-name=aq-pid-wikirag
#SBATCH --time=01:00:00
#SBATCH --mem=4GB
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

LOG_DIR="logs/slurm/privileged_information_distillation"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/wikipedia-rag-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

python experiments/privileged_information_distillation/build_wikipedia_rag_cache.py \
  --split validation \
  --scenario varied-deception \
  --output results/blackbox/privileged_information_distillation_rag/wikipedia_varied_validation.jsonl \
  --delay-seconds 1.0
