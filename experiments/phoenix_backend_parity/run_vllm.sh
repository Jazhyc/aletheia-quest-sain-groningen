#!/bin/bash
#SBATCH --job-name=phoenix-eunomia-vllm
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail

ROOT="/scratch/s4626451/Aletheias-Quest-Competition"
cd "${ROOT}"

LOG_DIR="logs/slurm/phoenix_backend_parity"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eunomia-vllm-${SLURM_JOB_ID}.out"
exec >"${LOG_FILE}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate

python experiments/phoenix_backend_parity/run.py --backend vllm

