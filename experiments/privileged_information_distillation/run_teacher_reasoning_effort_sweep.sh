#!/bin/bash
#SBATCH --job-name=aq-pid-effort-sweep
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/privileged_information_distillation"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/teacher-effort-sweep-${SLURM_JOB_ID}.out"
SERVER_LOG="${LOG_DIR}/teacher-effort-server-${SLURM_JOB_ID}.out"
exec >"${MAIN_LOG}" 2>&1
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
echo "server_ready api_base=${API_BASE}"

for effort in low medium high; do
  echo "direct_validation_effort=${effort}"
  python experiments/blackbox/run_judge.py \
    --config-name single_judges/blackbox_reasoning_gpt_oss_120b_nothink_truth_value_v1 \
    "method=gpt_oss_120b_truth_value_effort_${effort}_v1" \
    judge.backend=openai \
    "judge.api_base=${API_BASE}" \
    "judge.served_model=${SERVED_MODEL}" \
    judge.max_tokens=2048 \
    "judge.reasoning_effort=${effort}" \
    judge.request_timeout=600 \
    split=validation
done

for config_name in pid_teacher_effort_low_v1 pid_teacher_effort_high_v1; do
  echo "privileged_trace_config=${config_name}"
  python experiments/privileged_information_distillation/generate_teacher_data.py \
    --config-name "${config_name}" \
    teacher.backend=openai \
    "teacher.api_base=${API_BASE}" \
    "teacher.served_model=${SERVED_MODEL}" \
    teacher.request_timeout=600
done

echo "reasoning-effort teacher sweep complete"
