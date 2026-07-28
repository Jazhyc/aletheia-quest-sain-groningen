#!/bin/bash
# Submit the matched OpenRouter Qwen3.5-397B clarified-prompt soft student.

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

CONFIG="pid_qwen397_openrouter_explicit_tvg_binary_soft_distillation_v1"
METHOD="qwen9b_qwen397_openrouter_explicit_tvg_binary_softonly_varied_v1"
CACHE_ROOT="results/blackbox/qwen35_397b_openrouter_nothink_tvg_binary_logit_explicit_digits_v1/train"
ADAPTER="results/blackbox/${METHOD}/adapter"
MIGRATION_WORK_DIR="results/blackbox/${METHOD}/peft_path_migration"

sha256sum -c "${CACHE_ROOT}/SHA256SUMS"
for artifact in generations.jsonl soft_targets.jsonl student_rows.jsonl; do
  rows="$(wc -l < "${CACHE_ROOT}/${artifact}")"
  if [[ "${rows}" -ne 2880 ]]; then
    echo "expected 2880 rows in ${CACHE_ROOT}/${artifact}, found ${rows}" >&2
    exit 1
  fi
done

STUDENT="$(submit_id \
  --job-name=aq-q397-or-exp-soft \
  --time=01:00:00 \
  --export="ALL,QWEN35_CANONICALIZE_ADAPTER=${ROOT}/${ADAPTER},QWEN35_CANONICALIZATION_WORK_DIR=${ROOT}/${MIGRATION_WORK_DIR}" \
  "${ROOT}/experiments/privileged_information_distillation/run_student_sft.sh" \
  --config-name "${CONFIG}")"

VALIDATION="$(submit_id \
  --job-name=aq-q397-or-exp-eval \
  --time=01:00:00 \
  --dependency="afterok:${STUDENT}" \
  "${ROOT}/experiments/privileged_information_distillation/evaluate_student_sft.sh" \
  --adapter-dir "${ADAPTER}" \
  --split validation \
  --run-name validation_qwen397_openrouter_explicit_tvg_binary_soft_v1 \
  --max-new-tokens 1 \
  --continuous-margins \
  --verify-lora-effect \
  --continuous-margin-condition direct)"

printf 'student=%s\n' "${STUDENT}"
printf 'validation=%s\n' "${VALIDATION}"
