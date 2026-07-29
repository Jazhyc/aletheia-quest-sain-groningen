#!/bin/bash
# Calibrate finite-difference DataRater scores against exact soft-target gradients.

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

CACHE_ROOT="results/blackbox/qwen35_397b_fp8_nothink_truth_value_binary_logit_v1/train"
OUTPUT_ROOT="results/blackbox/qwen397_soft_datarater_calibration_v1"
COMMON_ARGS=(
  --input "${CACHE_ROOT}/student_rows.jsonl"
  --objective soft_binary
  --soft-teacher-artifact "${CACHE_ROOT}/soft_targets.jsonl"
  --dataset-name-contains varied-deception
  --keep-fractions 0.5
  --meta-fraction 0.05
  --max-meta-records 36
  --max-candidates 72
  --meta-batch-size 4
  --candidate-batch-size 8
  --lora-rank 16
  --last-layers 1
  --max-length 2048
)

sha256sum -c "${CACHE_ROOT}/SHA256SUMS"

EXACT="$(submit_id \
  --job-name=aq-q397-dr-exact \
  --time=01:00:00 \
  "${ROOT}/experiments/privileged_information_distillation/run_datarater_score.sh" \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/exact" \
  --scoring-mode exact)"

FD001="$(submit_id \
  --job-name=aq-q397-dr-fd001 \
  --time=01:00:00 \
  "${ROOT}/experiments/privileged_information_distillation/run_datarater_score.sh" \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/fd001" \
  --scoring-mode finite_difference \
  --finite-difference-epsilon 0.01)"

FD003="$(submit_id \
  --job-name=aq-q397-dr-fd003 \
  --time=01:00:00 \
  "${ROOT}/experiments/privileged_information_distillation/run_datarater_score.sh" \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/fd003" \
  --scoring-mode finite_difference \
  --finite-difference-epsilon 0.03)"

FD010="$(submit_id \
  --job-name=aq-q397-dr-fd010 \
  --time=01:00:00 \
  "${ROOT}/experiments/privileged_information_distillation/run_datarater_score.sh" \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/fd010" \
  --scoring-mode finite_difference \
  --finite-difference-epsilon 0.1)"

printf 'exact=%s\n' "${EXACT}"
printf 'fd001=%s\n' "${FD001}"
printf 'fd003=%s\n' "${FD003}"
printf 'fd010=%s\n' "${FD010}"
