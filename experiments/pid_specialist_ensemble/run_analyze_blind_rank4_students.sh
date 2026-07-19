#!/bin/bash
#SBATCH --job-name=aq-blind-r4-stack
#SBATCH --time=00:10:00
#SBATCH --mem=8GB
#SBATCH --partition=regularshort
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/pid_specialist_ensemble"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/blind-rank4-analyze-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

ENSEMBLE_DIR="results/blackbox/qwen9b_blind_teacher_ensemble_rank1_v1"
python experiments/pid_specialist_ensemble/analyze_ensemble.py \
  --train-member material=results/blackbox/qwen9b_blind_teacher_material_rank4_v1/train_meta_features_v1/generations.jsonl \
  --train-member polarity=results/blackbox/qwen9b_blind_teacher_polarity_rank4_v1/train_meta_features_v1/generations.jsonl \
  --train-member hierarchy=results/blackbox/qwen9b_blind_teacher_hierarchy_rank4_v1/train_meta_features_v1/generations.jsonl \
  --validation-member material=results/blackbox/qwen9b_blind_teacher_material_rank4_v1/validation_blind_rank4_ensemble_v1/generations.jsonl \
  --validation-member polarity=results/blackbox/qwen9b_blind_teacher_polarity_rank4_v1/validation_blind_rank4_ensemble_v1/generations.jsonl \
  --validation-member hierarchy=results/blackbox/qwen9b_blind_teacher_hierarchy_rank4_v1/validation_blind_rank4_ensemble_v1/generations.jsonl \
  --selection-manifest "${ENSEMBLE_DIR}/common_parsed_train.jsonl" \
  --output "${ENSEMBLE_DIR}/student_rank4_ensemble_result.json"

python experiments/pid_specialist_ensemble/analyze_blind_teachers.py \
  --member material=results/blackbox/qwen9b_blind_teacher_material_rank1_v1/teacher/validation.jsonl \
  --member polarity=results/blackbox/qwen9b_blind_teacher_polarity_rank1_v1/teacher/validation.jsonl \
  --member hierarchy=results/blackbox/qwen9b_blind_teacher_hierarchy_rank1_v1/teacher/validation.jsonl \
  --student-member material=results/blackbox/qwen9b_blind_teacher_material_rank4_v1/validation_blind_rank4_ensemble_v1/generations.jsonl \
  --student-member polarity=results/blackbox/qwen9b_blind_teacher_polarity_rank4_v1/validation_blind_rank4_ensemble_v1/generations.jsonl \
  --student-member hierarchy=results/blackbox/qwen9b_blind_teacher_hierarchy_rank4_v1/validation_blind_rank4_ensemble_v1/generations.jsonl \
  --output "${ENSEMBLE_DIR}/teacher_student_rank4_agreement.json"

python experiments/pid_specialist_ensemble/compare_student_ranks.py \
  --rank1-member material=results/blackbox/qwen9b_blind_teacher_material_rank1_v1/validation_blind_ensemble_v1/generations.jsonl \
  --rank1-member polarity=results/blackbox/qwen9b_blind_teacher_polarity_rank1_v1/validation_blind_ensemble_v1/generations.jsonl \
  --rank1-member hierarchy=results/blackbox/qwen9b_blind_teacher_hierarchy_rank1_v1/validation_blind_ensemble_v1/generations.jsonl \
  --rank4-member material=results/blackbox/qwen9b_blind_teacher_material_rank4_v1/validation_blind_rank4_ensemble_v1/generations.jsonl \
  --rank4-member polarity=results/blackbox/qwen9b_blind_teacher_polarity_rank4_v1/validation_blind_rank4_ensemble_v1/generations.jsonl \
  --rank4-member hierarchy=results/blackbox/qwen9b_blind_teacher_hierarchy_rank4_v1/validation_blind_rank4_ensemble_v1/generations.jsonl \
  --output "${ENSEMBLE_DIR}/student_rank1_rank4_comparison.json"
