#!/bin/bash
#SBATCH --job-name=aq-intent-sweep
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/heterogeneous_adapter_ensemble"
RESULT_DIR="results/blackbox/qwen9b_belief_intent_prompt_sweep_v1"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}/teacher"
exec >"${LOG_DIR}/intent-sweep-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

MODEL="openai/gpt-oss-120b"
SERVED_MODEL="gpt-oss-intent-teacher"
PORT="$((20000 + SLURM_JOB_ID % 10000))"
API_BASE="http://127.0.0.1:${PORT}/v1"
SERVER_LOG="${LOG_DIR}/intent-sweep-server-${SLURM_JOB_ID}.out"

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

declare -A CONFIGS=(
  [baseline]="pid_heterogeneous_resolved_intent_rank1_v1"
  [belief_statement]="pid_heterogeneous_belief_statement_rank1_v1"
  [belief_proof]="pid_heterogeneous_belief_proof_rank1_v1"
)

for name in baseline belief_statement belief_proof; do
  artifact="${RESULT_DIR}/teacher/${name}_full_train.jsonl"
  python experiments/privileged_information_distillation/generate_teacher_data.py \
    --config-name "${CONFIGS[${name}]}" \
    teacher.backend=openai \
    "teacher.api_base=${API_BASE}" \
    "teacher.served_model=${SERVED_MODEL}" \
    teacher.request_timeout=600 \
    teacher.selection_manifest=null \
    "teacher.artifact=${artifact}"
done

python experiments/heterogeneous_adapter_ensemble/analyze_intent_teacher_sweep.py \
  --member baseline="${RESULT_DIR}/teacher/baseline_full_train.jsonl" \
  --member belief_statement="${RESULT_DIR}/teacher/belief_statement_full_train.jsonl" \
  --member belief_proof="${RESULT_DIR}/teacher/belief_proof_full_train.jsonl" \
  --output "${RESULT_DIR}/full_train_result.json"
