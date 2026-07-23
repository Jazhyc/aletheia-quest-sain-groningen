#!/bin/bash
#SBATCH --job-name=aq-wikidata-sweep
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
exec >"${LOG_DIR}/judge-sweep-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

CACHE="${CACHE:-results/blackbox/wikidata_rag_daily_v1/validation_sweep_cache.jsonl}"
OUTPUT="${OUTPUT:-results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/validation_wikidata_3condition_v1}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  python experiments/wikidata_rag/build_validation_cache.py \
    --database results/blackbox/wikidata_rag_daily_v1/wikidata.sqlite \
    --output "${CACHE}" \
    --split validation
fi

python experiments/wikidata_rag/evaluate_judge_sweep.py \
  --adapter-dir results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter \
  --cache "${CACHE}" \
  --output-dir "${OUTPUT}" \
  --split validation
