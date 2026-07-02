#!/usr/bin/env bash
#SBATCH --job-name=aq-blackbox
#SBATCH --output=slurm-%x-%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if command -v module >/dev/null 2>&1; then
  module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.2.0 || true
fi

source .venv/bin/activate

python scripts/run_blackbox_judge.py "$@"
