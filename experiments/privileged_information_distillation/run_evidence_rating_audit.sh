#!/bin/bash
#SBATCH --job-name=aq-rating-audit
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --partition=regularshort
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/privileged_information_distillation
LOG="logs/slurm/privileged_information_distillation/rating-audit-${SLURM_JOB_ID}.out"
exec >"${LOG}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

python experiments/privileged_information_distillation/audit_evidence_rating_cache.py "$@"
