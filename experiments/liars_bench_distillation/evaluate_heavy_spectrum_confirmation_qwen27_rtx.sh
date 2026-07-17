#!/usr/bin/env bash
#SBATCH --job-name=lb-confirm-q27-rtx
#SBATCH --partition=gpushort
#SBATCH --nodes=1
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail

cd /scratch/s4626451/Aletheias-Quest-Competition
mkdir -p logs/slurm/liars_bench_distillation
bootstrap="logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"
target="logs/slurm/liars_bench_distillation/q27-confirm-rtx-${SLURM_JOB_ID}.out"
exec > >(tee -a "$target") 2>&1
rm -f "$bootstrap"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

python experiments/liars_bench_distillation/evaluate_heavy_spectrum.py \
  --eval-artifact results/blackbox/liars_bench_heavy_spectrum_confirmation_v1/eval.jsonl \
  --judge-config results/blackbox/qwen27b_reason_ensemble_dks_member4096_v1/validation/config.yaml \
  --output-dir results/blackbox/liars_bench_qwen27_heavy_spectrum_confirmation_v1
