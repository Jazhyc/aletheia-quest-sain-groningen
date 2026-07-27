#!/bin/bash
#SBATCH --job-name=aq-qwen-ret-r1
#SBATCH --time=01:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/wikidata_rag
LOG="logs/slurm/wikidata_rag/qwen-retriever-rank1-${SLURM_JOB_ID}.out"
exec >"${LOG}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

METHOD_DIR="results/blackbox/wikidata_qwen_retriever_rank1_v1"
mkdir -p "${METHOD_DIR}/teacher"
python -m experiments.wikidata_rag.build_qwen_retriever_distillation \
  --teacher-labels results/blackbox/wikidata_rag_gptoss_supervision_v1/train_labels_decisive_full_v1.jsonl \
  --planner-output results/blackbox/wikidata_rag_qwen_planner_v1/train_full_v1.jsonl \
  --output "${METHOD_DIR}/teacher/train_pairs.jsonl" \
  --report "${METHOD_DIR}/teacher/train_pairs_report.json"

python experiments/privileged_information_distillation/train_student_sft.py \
  --config-name wikidata_qwen_retriever_rank1_v1
