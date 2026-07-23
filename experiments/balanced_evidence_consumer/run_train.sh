#!/bin/bash
#SBATCH --job-name=aq-balanced-evidence
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/balanced_evidence_consumer
LOG="logs/slurm/balanced_evidence_consumer/train-${SLURM_JOB_ID}.out"
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

OUTPUT="results/blackbox/qwen9b_fever_balanced_evidence_consumer_rank1_v1/curriculum/train.jsonl"
REPORT="results/blackbox/qwen9b_fever_balanced_evidence_consumer_rank1_v1/curriculum/audit.json"
python experiments/balanced_evidence_consumer/build_curriculum.py \
  --baseline-teacher results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl \
  --real-teacher results/blackbox/qwen9b_privileged_gptoss120b_fever_visible_real_variedonly_adamw5e5_v1/teacher/train.jsonl \
  --shuffled-teacher results/blackbox/qwen9b_privileged_gptoss120b_fever_visible_shuffled_variedonly_adamw5e5_v1/teacher/train.jsonl \
  --audits results/blackbox/fever_fact_verification_train_v1/varied_train_selective_initial_audit.jsonl \
  --output "${OUTPUT}" \
  --report "${REPORT}"

python experiments/privileged_information_distillation/train_student_sft.py \
  --config-name privileged_information_distillation_fever_balanced_rank1 "$@"
