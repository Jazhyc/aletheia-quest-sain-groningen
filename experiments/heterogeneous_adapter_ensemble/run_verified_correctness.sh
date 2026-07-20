#!/bin/bash
#SBATCH --job-name=aq-verified-correctness
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/heterogeneous_adapter_ensemble"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/verified-correctness-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

METHOD_DIR="results/blackbox/qwen9b_heterogeneous_verified_correctness_rank1_v1"
CURRICULUM_DIR="${METHOD_DIR}/curriculum"

python -m experiments.wikidata_rag.build_grounded_synthetic_supervision \
  --entity-database results/blackbox/wikidata_rag_broad_v1/wikidata.sqlite \
  --output "${CURRICULUM_DIR}/raw.jsonl" \
  --rows 4000 \
  --candidate-count 8 \
  --seed 42

python -m experiments.heterogeneous_adapter_ensemble.build_verified_correctness_curriculum \
  --input "${CURRICULUM_DIR}/raw.jsonl" \
  --train-output "${CURRICULUM_DIR}/train.jsonl" \
  --validation-output "${CURRICULUM_DIR}/validation.jsonl" \
  --train-per-negative 200 \
  --validation-per-negative 50 \
  --seed 42

python experiments/privileged_information_distillation/train_student_sft.py \
  --config-name pid_heterogeneous_verified_correctness_rank1_v1

python experiments/wikidata_rag/evaluate_matched_reader.py \
  --adapter-dir "${METHOD_DIR}/adapter" \
  --adapter-dir results/blackbox/qwen9b_heterogeneous_incorrectness_rank1_v1/adapter \
  --retrieval-cache results/blackbox/fever_fact_verification_v1/varied_validation_selective_expanded_window_union_reference_v1.jsonl \
  --correctness-validation-cache "${CURRICULUM_DIR}/validation.jsonl" \
  --condition empty \
  --condition real \
  --condition shuffled \
  --run-name validation_verified_correctness_v1 \
  --max-new-tokens 512 \
  --max-model-len 4096
