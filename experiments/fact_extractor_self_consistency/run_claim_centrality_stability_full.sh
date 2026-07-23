#!/bin/bash
#SBATCH --job-name=aq-claim-stability-full
#SBATCH --time=01:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if command -v module >/dev/null 2>&1; then
  module load Python/3.12.3-GCCcore-13.3.0
  module load CUDA/13.2.0
fi
source .venv/bin/activate
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

METHOD="fact_extractor_self_consistency"
LOG_DIR="logs/slurm/${METHOD}"
mkdir -p "${LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${LOG_DIR}/claim-stability-full-validation-${SLURM_JOB_ID}.out" 2>&1
  rm -f "logs/slurm/${SLURM_JOB_NAME:-aq-claim-stability-full}-${SLURM_JOB_ID}.bootstrap.out"
  echo "job_id=${SLURM_JOB_ID}"
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

python experiments/fact_extractor_self_consistency/run_claim_centrality_stability_pilot.py extract --scope full
python experiments/fact_extractor_self_consistency/run_claim_centrality_stability_pilot.py verify --scope full
python experiments/fact_extractor_self_consistency/run_claim_centrality_stability_pilot.py summarize --scope full
