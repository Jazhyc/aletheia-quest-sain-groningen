#!/bin/bash
#SBATCH --job-name=aq-qwen-consumers
#SBATCH --time=00:45:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/wikidata_rag
LOG="logs/slurm/wikidata_rag/qwen-planner-consumers-${SLURM_JOB_ID}.out"
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

python -m experiments.wikidata_rag.evaluate_qwen_planner_consumers \
  --consumer matched_wikidata=results/blackbox/qwen9b_pid_wikidata_matched_reader_v1/adapter \
  --consumer fever_visible=results/blackbox/qwen9b_privileged_gptoss120b_fever_visible_real_variedonly_adamw5e5_v1/adapter \
  --cache unfiltered=results/blackbox/wikidata_rag_qwen_planner_v1/validation_sweep_cache_v1.jsonl \
  --cache filtered=results/blackbox/wikidata_qwen_retriever_rank1_v1/evaluation/validation_filtered_base_sweep_cache.jsonl \
  --output-dir results/blackbox/wikidata_qwen_planner_consumer_crossover_v1
