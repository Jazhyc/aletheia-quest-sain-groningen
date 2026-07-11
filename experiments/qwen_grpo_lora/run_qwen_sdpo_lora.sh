#!/bin/bash
#SBATCH --job-name=aq-qwen-sdpo
#SBATCH --time=08:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if command -v module >/dev/null 2>&1; then
  module load Python/3.12.3-GCCcore-13.3.0
  module load CUDA/13.2.0
fi

source /scratch/s4626451/.venvs/aletheia-sdpo/bin/activate
METHOD="qwen9b_pid_varied_sdpo_live_v1"
for arg in "$@"; do
  if [[ "${arg}" == method=* ]]; then METHOD="${arg#method=}"; fi
done
LOG_DIR="logs/slurm/${METHOD}"
mkdir -p "${LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${LOG_DIR}/train-${SLURM_JOB_ID}.out" 2>&1
  rm -f "logs/slurm/${SLURM_JOB_NAME:-aq-qwen-sdpo}-${SLURM_JOB_ID}.bootstrap.out"
  echo "job_id=${SLURM_JOB_ID}"
fi

if [[ -f .env ]]; then set -a; source .env; set +a; fi
export TOKENIZERS_PARALLELISM=false
export TRL_EXPERIMENTAL_SILENCE=1
export WANDB_DIR="${WANDB_DIR:-logs/wandb}"
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 50000))}"
mkdir -p "${WANDB_DIR}"

accelerate launch --num_processes 1 --main_process_port "${MASTER_PORT}" \
  experiments/qwen_grpo_lora/run_qwen_grpo_lora.py \
  --config-name qwen_sdpo_lora_pid_varied "$@"
