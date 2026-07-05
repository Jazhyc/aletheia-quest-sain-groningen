#!/bin/bash
#SBATCH --job-name=aq-qwen-grpo
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

source .venv/bin/activate

METHOD="qwen_grpo_lora_r1_nothink_v1"
METHOD_LOG_DIR="logs/slurm/${METHOD}"
mkdir -p "${METHOD_LOG_DIR}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  METHOD_LOG_FILE="${METHOD_LOG_DIR}/train-${SLURM_JOB_ID}.out"
  BOOTSTRAP_LOG_FILE="logs/slurm/${SLURM_JOB_NAME:-aq-qwen-grpo}-${SLURM_JOB_ID}.bootstrap.out"
  echo "Redirecting Slurm job output to ${METHOD_LOG_FILE}"
  exec >"${METHOD_LOG_FILE}" 2>&1
  rm -f "${BOOTSTRAP_LOG_FILE}"
  echo "job_id=${SLURM_JOB_ID}"
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="${WANDB_DIR:-logs/wandb}"
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-/scratch/${USER}}/.cache/uv}"
mkdir -p "${WANDB_DIR}"

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_METRICS_FILE="${METHOD_LOG_DIR}/gpu-${SLURM_JOB_ID}.csv"
  (
    echo "timestamp,index,utilization.gpu [%],utilization.memory [%],memory.used [MiB],memory.total [MiB],power.draw [W]"
    while true; do
      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits
      sleep 5
    done
  ) >"${GPU_METRICS_FILE}" 2>/dev/null &
  GPU_MONITOR_PID=$!
  trap 'kill "${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT
  echo "gpu_metrics=${GPU_METRICS_FILE}"
fi

python - <<'PY'
import importlib.util
missing = [pkg for pkg in ("trl", "peft", "wandb") if importlib.util.find_spec(pkg) is None]
if missing:
    raise SystemExit(
        "Missing training dependencies: "
        + ", ".join(missing)
        + ". Run './setup_dev.sh' or 'uv sync' after the pyproject dependency update."
    )
PY

accelerate launch --num_processes 1 experiments/qwen_grpo_lora/run_qwen_grpo_lora.py "$@"
