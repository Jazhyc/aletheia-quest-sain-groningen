#!/bin/bash
# Submit matched standardized-BCE and Huber Qwen3.5-397B distillation arms.

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
sha256sum -c "${CACHE_ROOT}/SHA256SUMS"
for artifact in generations.jsonl soft_targets.jsonl student_rows.jsonl; do
  rows="$(wc -l < "${CACHE_ROOT}/${artifact}")"
  if [[ "${rows}" -ne 2880 ]]; then
    echo "expected 2880 rows in ${CACHE_ROOT}/${artifact}, found ${rows}" >&2
    exit 1
  fi
done

submit_arm() {
  local arm="$1"
  local config="$2"
  local method="$3"
  local adapter="results/blackbox/${method}/adapter"
  local migration="results/blackbox/${method}/peft_path_migration"
  local student
  local validation

  student="$(submit_id \
    --job-name="aq-q397-${arm}-student" \
    --time=01:00:00 \
    --export="ALL,QWEN35_CANONICALIZE_ADAPTER=${ROOT}/${adapter},QWEN35_CANONICALIZATION_WORK_DIR=${ROOT}/${migration}" \
    "${ROOT}/experiments/privileged_information_distillation/run_student_sft.sh" \
    --config-name "${config}")"

  validation="$(submit_id \
    --job-name="aq-q397-${arm}-eval" \
    --time=01:00:00 \
    --dependency="afterok:${student}" \
    "${ROOT}/experiments/privileged_information_distillation/evaluate_student_sft.sh" \
    --adapter-dir "${adapter}" \
    --split validation \
    --run-name "validation_qwen397_tvg_binary_${arm}_v1" \
    --max-new-tokens 1 \
    --continuous-margins \
    --verify-lora-effect \
    --continuous-margin-condition direct)"

  printf '%s_student=%s\n' "${arm}" "${student}"
  printf '%s_validation=%s\n' "${arm}" "${validation}"
}

submit_arm \
  "zscorebce" \
  "pid_qwen397_tvg_binary_soft_distillation_zscore_bce_v1" \
  "qwen9b_qwen397_tvg_binary_zscorebce_varied_v1"
submit_arm \
  "zscorehuber" \
  "pid_qwen397_tvg_binary_soft_distillation_zscore_huber_v1" \
  "qwen9b_qwen397_tvg_binary_zscorehuber_varied_v1"
