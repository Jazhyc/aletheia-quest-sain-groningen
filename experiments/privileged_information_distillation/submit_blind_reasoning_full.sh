#!/bin/bash
# Submit full varied-data blind GPT-OSS rationale generation, rank-16 SFT, and validation.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "run this from a project checkout containing .venv" >&2
  exit 1
fi

submit_id() {
  local submitted
  submitted="$(sbatch --parsable "$@")"
  printf '%s' "${submitted%%;*}"
}

CONFIG="pid_blind_teacher_material_rank16_full_v1"
METHOD="qwen9b_blind_gptoss120b_material_reasoning_full_r16_v1"
ADAPTER="results/blackbox/${METHOD}/adapter"

TEACHER="$(submit_id \
  --job-name=aq-blind-full-teacher \
  --time=02:00:00 \
  "${ROOT}/experiments/privileged_information_distillation/run_teacher.sh" \
  --config-name "${CONFIG}")"

STUDENT="$(submit_id \
  --job-name=aq-blind-full-student \
  --time=02:00:00 \
  --dependency="afterok:${TEACHER}" \
  "${ROOT}/experiments/privileged_information_distillation/run_student_sft.sh" \
  --config-name "${CONFIG}")"

VALIDATION="$(submit_id \
  --job-name=aq-blind-full-eval \
  --time=01:00:00 \
  --dependency="afterok:${STUDENT}" \
  "${ROOT}/experiments/privileged_information_distillation/evaluate_student_sft.sh" \
  --adapter-dir "${ADAPTER}" \
  --split validation \
  --run-name validation_blind_reasoning_full_r16_v1 \
  --max-new-tokens 512 \
  --continuous-margins \
  --continuous-margin-condition direct)"

printf 'teacher=%s\n' "${TEACHER}"
printf 'student=%s\n' "${STUDENT}"
printf 'validation=%s\n' "${VALIDATION}"
