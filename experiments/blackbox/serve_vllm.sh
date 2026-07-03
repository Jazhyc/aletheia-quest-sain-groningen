#!/bin/bash

set -euo pipefail

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0

source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL="${SERVED_MODEL:-qwen-judge}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_LOGPROBS="${MAX_LOGPROBS:-20}"
LOG_DIR="${LOG_DIR:-logs/vllm}"

mkdir -p "${LOG_DIR}"

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-logprobs "${MAX_LOGPROBS}" \
  --enable-prefix-caching
