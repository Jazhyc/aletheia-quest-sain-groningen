#!/bin/bash
#SBATCH --job-name=aq-hetero-targets
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/heterogeneous_adapter_ensemble"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/targets-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

MODEL="openai/gpt-oss-120b"
SERVED_MODEL="gpt-oss-heterogeneous-teacher"
PORT="$((20000 + SLURM_JOB_ID % 10000))"
API_BASE="http://127.0.0.1:${PORT}/v1"
SERVER_LOG="${LOG_DIR}/target-server-${SLURM_JOB_ID}.out"

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

for objective in incorrectness resolved_intent; do
  config="pid_heterogeneous_${objective}_rank1_v1"
  output="results/blackbox/qwen9b_heterogeneous_${objective}_rank1_v1/teacher"
  python experiments/privileged_information_distillation/generate_teacher_data.py \
    --config-name "${config}" \
    teacher.backend=openai \
    "teacher.api_base=${API_BASE}" \
    "teacher.served_model=${SERVED_MODEL}" \
    teacher.request_timeout=600
  python experiments/privileged_information_distillation/generate_teacher_data.py \
    --config-name "${config}" \
    teacher.backend=openai \
    "teacher.api_base=${API_BASE}" \
    "teacher.served_model=${SERVED_MODEL}" \
    teacher.request_timeout=600 \
    teacher.split=validation \
    teacher.selection_manifest=null \
    "teacher.artifact=${output}/validation.jsonl"
done

ENSEMBLE_DIR="results/blackbox/qwen9b_heterogeneous_adapter_ensemble_rank1_v1"
for objective in incorrectness resolved_intent; do
  python experiments/pid_specialist_ensemble/prepare_common_manifest.py \
    --cache "results/blackbox/qwen9b_heterogeneous_${objective}_rank1_v1/teacher/train.jsonl" \
    --fraction 1.0 \
    --seed 0 \
    --allow-label-mismatch \
    --output "${ENSEMBLE_DIR}/${objective}_parsed_train.jsonl"
done

python experiments/pid_specialist_ensemble/analyze_blind_teachers.py \
  --member incorrectness=results/blackbox/qwen9b_heterogeneous_incorrectness_rank1_v1/teacher/validation.jsonl \
  --member resolved_intent=results/blackbox/qwen9b_heterogeneous_resolved_intent_rank1_v1/teacher/validation.jsonl \
  --output "${ENSEMBLE_DIR}/teacher_validation_result.json"
