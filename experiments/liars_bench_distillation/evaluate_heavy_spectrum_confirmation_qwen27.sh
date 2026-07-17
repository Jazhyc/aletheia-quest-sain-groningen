#!/bin/bash
#SBATCH --job-name=aq-q27-heavy-confirm
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm/liars_bench_distillation
exec >"logs/slurm/liars_bench_distillation/q27-heavy-confirm-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

python experiments/liars_bench_distillation/evaluate_heavy_spectrum.py \
  --eval-artifact results/blackbox/liars_bench_heavy_spectrum_confirmation_v1/eval.jsonl \
  --judge-config results/blackbox/qwen27b_reason_ensemble_dks_member4096_v1/validation/config.yaml \
  --tensor-parallel-size 2 \
  --output-dir results/blackbox/liars_bench_qwen27_heavy_spectrum_confirmation_v1
