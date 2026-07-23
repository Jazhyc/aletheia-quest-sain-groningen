#!/bin/bash
#SBATCH --job-name=aq-pid-r1-specialist-stack
#SBATCH --time=00:10:00
#SBATCH --mem=8GB
#SBATCH --partition=regularshort
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/pid_specialist_ensemble"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/analyze-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

python experiments/pid_specialist_ensemble/analyze_ensemble.py \
  --train-member material=results/blackbox/qwen9b_pid_specialist_material_rank1_v1/train_meta_features_v1/generations.jsonl \
  --train-member polarity=results/blackbox/qwen9b_pid_specialist_polarity_rank1_v1/train_meta_features_v1/generations.jsonl \
  --train-member hierarchy=results/blackbox/qwen9b_pid_specialist_hierarchy_rank1_v1/train_meta_features_v1/generations.jsonl \
  --validation-member material=results/blackbox/qwen9b_pid_specialist_material_rank1_v1/validation_specialist_ensemble_v1/generations.jsonl \
  --validation-member polarity=results/blackbox/qwen9b_pid_specialist_polarity_rank1_v1/validation_specialist_ensemble_v1/generations.jsonl \
  --validation-member hierarchy=results/blackbox/qwen9b_pid_specialist_hierarchy_rank1_v1/validation_specialist_ensemble_v1/generations.jsonl \
  --selection-manifest results/blackbox/qwen9b_pid_specialist_ensemble_rank1_v1/common_train10.jsonl \
  --output results/blackbox/qwen9b_pid_specialist_ensemble_rank1_v1/ensemble_result.json
