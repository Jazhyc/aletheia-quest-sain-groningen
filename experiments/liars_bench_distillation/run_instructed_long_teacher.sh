#!/bin/bash
#SBATCH --job-name=aq-liars-long-teacher
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/liars_bench_distillation"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/instructed-long-teacher-${SLURM_JOB_ID}.out"
SERVER_LOG="${LOG_DIR}/instructed-long-server-${SLURM_JOB_ID}.out"
exec >"${LOG}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

MODEL="openai/gpt-oss-120b"
SERVED_MODEL="gpt-oss-teacher"
PORT="$((20000 + SLURM_JOB_ID % 10000))"
API_BASE="http://127.0.0.1:${PORT}/v1"

vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 240); do
  if curl -fsS "${API_BASE}/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM server exited during startup; see ${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS "${API_BASE}/models" >/dev/null

python experiments/liars_bench_distillation/prepare_teacher_data.py \
  --api-base "${API_BASE}" \
  --served-model "${SERVED_MODEL}" \
  --artifact results/blackbox/liars_bench_instructed_long_pid_v1/teacher/train.jsonl \
  --eval-artifact results/blackbox/liars_bench_instructed_long_pid_v1/eval.jsonl \
  --categories instructed-deception \
  --per-label-train 40 \
  --per-label-eval 100 \
  --minimum-output-chars 180 \
  --minimum-output-sentences 2 \
  --reasoning-effort medium

python experiments/liars_bench_distillation/audit_teacher_cache.py \
  results/blackbox/liars_bench_instructed_long_pid_v1/teacher/train.jsonl \
  --output results/blackbox/liars_bench_instructed_long_pid_v1/teacher/audit.json \
  --expected-total 80 \
  --expected-datasets 1
