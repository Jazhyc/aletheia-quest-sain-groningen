#!/bin/bash
#SBATCH --job-name=aq-blind-teachers
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

LOG_DIR="logs/slurm/pid_specialist_ensemble"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/blind-teachers-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

MODEL="openai/gpt-oss-120b"
SERVED_MODEL="gpt-oss-blind-teacher"
PORT="$((20000 + SLURM_JOB_ID % 10000))"
API_BASE="http://127.0.0.1:${PORT}/v1"
SERVER_LOG="${LOG_DIR}/blind-teacher-server-${SLURM_JOB_ID}.out"

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

for lens in material polarity hierarchy; do
  config="pid_blind_teacher_${lens}_rank1_v1"
  output="results/blackbox/qwen9b_blind_teacher_${lens}_rank1_v1/teacher"
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

ENSEMBLE_DIR="results/blackbox/qwen9b_blind_teacher_ensemble_rank1_v1"
python experiments/pid_specialist_ensemble/prepare_common_manifest.py \
  --cache results/blackbox/qwen9b_blind_teacher_material_rank1_v1/teacher/train.jsonl \
  --cache results/blackbox/qwen9b_blind_teacher_polarity_rank1_v1/teacher/train.jsonl \
  --cache results/blackbox/qwen9b_blind_teacher_hierarchy_rank1_v1/teacher/train.jsonl \
  --fraction 1.0 \
  --seed 0 \
  --allow-label-mismatch \
  --output "${ENSEMBLE_DIR}/common_parsed_train.jsonl"

python experiments/pid_specialist_ensemble/analyze_blind_teachers.py \
  --member material=results/blackbox/qwen9b_blind_teacher_material_rank1_v1/teacher/train.jsonl \
  --member polarity=results/blackbox/qwen9b_blind_teacher_polarity_rank1_v1/teacher/train.jsonl \
  --member hierarchy=results/blackbox/qwen9b_blind_teacher_hierarchy_rank1_v1/teacher/train.jsonl \
  --output "${ENSEMBLE_DIR}/teacher_train_result.json"
python experiments/pid_specialist_ensemble/analyze_blind_teachers.py \
  --member material=results/blackbox/qwen9b_blind_teacher_material_rank1_v1/teacher/validation.jsonl \
  --member polarity=results/blackbox/qwen9b_blind_teacher_polarity_rank1_v1/teacher/validation.jsonl \
  --member hierarchy=results/blackbox/qwen9b_blind_teacher_hierarchy_rank1_v1/teacher/validation.jsonl \
  --output "${ENSEMBLE_DIR}/teacher_validation_result.json"
