#!/bin/bash
# Submit the matched Qwen3.5-397B binary Truth Value Guard soft student and validation.

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

CONFIG="pid_qwen397_tvg_binary_soft_distillation_v1"
METHOD="qwen9b_qwen397_tvg_binary_softonly_varied_v1"
CACHE_ROOT="results/blackbox/qwen35_397b_fp8_nothink_truth_value_binary_logit_v1/train"
ADAPTER="results/blackbox/${METHOD}/adapter"

sha256sum -c "${CACHE_ROOT}/SHA256SUMS"
for artifact in generations.jsonl soft_targets.jsonl student_rows.jsonl; do
  rows="$(wc -l < "${CACHE_ROOT}/${artifact}")"
  if [[ "${rows}" -ne 2880 ]]; then
    echo "expected 2880 rows in ${CACHE_ROOT}/${artifact}, found ${rows}" >&2
    exit 1
  fi
done

STUDENT="$(submit_id \
  --job-name=aq-q397-soft-student \
  --time=01:00:00 \
  "${ROOT}/experiments/privileged_information_distillation/run_student_sft.sh" \
  --config-name "${CONFIG}")"

VALIDATION="$(submit_id \
  --job-name=aq-q397-soft-eval \
  --time=01:00:00 \
  --dependency="afterok:${STUDENT}" \
  "${ROOT}/experiments/privileged_information_distillation/evaluate_student_sft.sh" \
  --adapter-dir "${ADAPTER}" \
  --split validation \
  --run-name validation_qwen397_tvg_binary_soft_v1 \
  --max-new-tokens 1 \
  --continuous-margins \
  --continuous-margin-condition direct)"

printf 'student=%s\n' "${STUDENT}"
printf 'validation=%s\n' "${VALIDATION}"
