#!/bin/bash
#SBATCH --job-name=aq-hallucination-teacher
#SBATCH --time=01:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/hallucination_negative_distillation"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/teacher-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

METHOD="hallucination_negative_distillation_v1"
ROOT="results/blackbox/${METHOD}"
python experiments/hallucination_negative_distillation/screen_alpaca.py \
  --adapter results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter \
  --signature-artifact results/blackbox/liars_bench_frozen_judge_signatures_v1/phoenix_v21_adapter.jsonl \
  --screen-artifact "${ROOT}/screen.jsonl" \
  --candidate-artifact "${ROOT}/candidates.jsonl" \
  --eval-artifact "${ROOT}/eval.jsonl"

MODEL="openai/gpt-oss-120b"
SERVED_MODEL="gpt-oss-teacher"
PORT="$((20000 + SLURM_JOB_ID % 10000))"
API_BASE="http://127.0.0.1:${PORT}/v1"
SERVER_LOG="${LOG_DIR}/server-${SLURM_JOB_ID}.out"

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

python experiments/hallucination_negative_distillation/generate_teacher.py \
  --candidate-artifact "${ROOT}/candidates.jsonl" \
  --artifact "${ROOT}/teacher/train.jsonl" \
  --api-base "${API_BASE}" \
  --served-model "${SERVED_MODEL}" \
  --reasoning-effort medium

python experiments/liars_bench_distillation/audit_teacher_cache.py \
  "${ROOT}/teacher/train.jsonl" \
  --output "${ROOT}/teacher/audit.json" \
  --expected-datasets 1 \
  --max-label-imbalance-fraction 1.0
