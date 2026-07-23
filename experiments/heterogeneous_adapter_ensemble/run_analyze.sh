#!/bin/bash
#SBATCH --job-name=aq-hetero-stack
#SBATCH --time=00:10:00
#SBATCH --mem=8GB
#SBATCH --partition=regularshort
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/heterogeneous_adapter_ensemble"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/analyze-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

ENSEMBLE_DIR="results/blackbox/qwen9b_heterogeneous_adapter_ensemble_rank1_v1"
python experiments/pid_specialist_ensemble/analyze_ensemble.py \
  --train-member deception=results/blackbox/qwen9b_pid_specialist_material_rank1_v1/train_heterogeneous_objectives_v1/generations.jsonl \
  --train-member incorrectness=results/blackbox/qwen9b_heterogeneous_incorrectness_rank1_v1/train_heterogeneous_objectives_v1/generations.jsonl \
  --train-member resolved_intent=results/blackbox/qwen9b_heterogeneous_resolved_intent_rank1_v1/train_heterogeneous_objectives_v1/generations.jsonl \
  --validation-member deception=results/blackbox/qwen9b_pid_specialist_material_rank1_v1/validation_heterogeneous_objectives_v1/generations.jsonl \
  --validation-member incorrectness=results/blackbox/qwen9b_heterogeneous_incorrectness_rank1_v1/validation_heterogeneous_objectives_v1/generations.jsonl \
  --validation-member resolved_intent=results/blackbox/qwen9b_heterogeneous_resolved_intent_rank1_v1/validation_heterogeneous_objectives_v1/generations.jsonl \
  --selection-manifest results/blackbox/qwen9b_pid_specialist_ensemble_rank1_v1/common_train10.jsonl \
  --selection-manifest "${ENSEMBLE_DIR}/incorrectness_parsed_train.jsonl" \
  --selection-manifest "${ENSEMBLE_DIR}/resolved_intent_parsed_train.jsonl" \
  --output "${ENSEMBLE_DIR}/student_ensemble_result.json"

python experiments/pid_specialist_ensemble/analyze_blind_teachers.py \
  --member incorrectness=results/blackbox/qwen9b_heterogeneous_incorrectness_rank1_v1/teacher/validation.jsonl \
  --member resolved_intent=results/blackbox/qwen9b_heterogeneous_resolved_intent_rank1_v1/teacher/validation.jsonl \
  --student-member incorrectness=results/blackbox/qwen9b_heterogeneous_incorrectness_rank1_v1/validation_heterogeneous_objectives_v1/generations.jsonl \
  --student-member resolved_intent=results/blackbox/qwen9b_heterogeneous_resolved_intent_rank1_v1/validation_heterogeneous_objectives_v1/generations.jsonl \
  --output "${ENSEMBLE_DIR}/teacher_student_agreement.json"
