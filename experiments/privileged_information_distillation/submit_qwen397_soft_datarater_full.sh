#!/bin/bash
# Score, train, and validate the frozen Qwen-397B soft-DataRater 50% screen.

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
SCORE_ROOT="results/blackbox/qwen397_soft_datarater_rank16_last1_v1"
BASELINE="results/blackbox/qwen9b_qwen397_tvg_binary_softonly_varied_v1/adapter"

sha256sum -c "${CACHE_ROOT}/SHA256SUMS"

SCORE="$(submit_id \
  --job-name=aq-q397-dr-score \
  --time=02:00:00 \
  "${ROOT}/experiments/privileged_information_distillation/run_datarater_score.sh" \
  --input "${CACHE_ROOT}/student_rows.jsonl" \
  --output-dir "${SCORE_ROOT}" \
  --objective soft_binary \
  --soft-teacher-artifact "${CACHE_ROOT}/soft_targets.jsonl" \
  --dataset-name-contains varied-deception \
  --keep-fractions 0.5 \
  --meta-fraction 0.05 \
  --meta-batch-size 4 \
  --candidate-batch-size 8 \
  --lora-rank 16 \
  --last-layers 1 \
  --max-length 2048 \
  --scoring-mode finite_difference \
  --finite-difference-epsilon 0.1)"

submit_student() {
  local arm="$1"
  local config="$2"
  local method="$3"
  local adapter="results/blackbox/${method}/adapter"
  local migration="results/blackbox/${method}/peft_path_migration"
  local job_id

  job_id="$(submit_id \
    --job-name="aq-q397-dr-${arm}" \
    --time=01:00:00 \
    --dependency="afterok:${SCORE}" \
    --export="ALL,QWEN35_CANONICALIZE_ADAPTER=${ROOT}/${adapter},QWEN35_CANONICALIZATION_WORK_DIR=${ROOT}/${migration}" \
    "${ROOT}/experiments/privileged_information_distillation/run_student_sft.sh" \
    --config-name "${config}")"
  printf '%s' "${job_id}"
}

RANDOM="$(submit_student \
  random50 \
  pid_qwen397_soft_datarater_random50_fixed90_v1 \
  qwen9b_qwen397_soft_datarater_random50_fixed90_v1)"
LOSS="$(submit_student \
  loss50 \
  pid_qwen397_soft_datarater_loss50_fixed90_v1 \
  qwen9b_qwen397_soft_datarater_loss50_fixed90_v1)"
DOT="$(submit_student \
  dot50 \
  pid_qwen397_soft_datarater_dot50_fixed90_v1 \
  qwen9b_qwen397_soft_datarater_dot50_fixed90_v1)"

RANDOM_ADAPTER="results/blackbox/qwen9b_qwen397_soft_datarater_random50_fixed90_v1/adapter"
LOSS_ADAPTER="results/blackbox/qwen9b_qwen397_soft_datarater_loss50_fixed90_v1/adapter"
DOT_ADAPTER="results/blackbox/qwen9b_qwen397_soft_datarater_dot50_fixed90_v1/adapter"
TRAIN_DEPENDENCY="afterok:${RANDOM}:${LOSS}:${DOT}"

FORWARD="$(submit_id \
  --job-name=aq-q397-dr-eval-fwd \
  --time=01:00:00 \
  --dependency="${TRAIN_DEPENDENCY}" \
  "${ROOT}/experiments/privileged_information_distillation/evaluate_student_sft.sh" \
  --adapter-dir "${BASELINE}" \
  --adapter-dir "${RANDOM_ADAPTER}" \
  --adapter-dir "${LOSS_ADAPTER}" \
  --adapter-dir "${DOT_ADAPTER}" \
  --split validation \
  --run-name validation_datarater_forward_v1 \
  --max-new-tokens 1 \
  --continuous-margins \
  --continuous-margin-condition direct)"

REVERSE="$(submit_id \
  --job-name=aq-q397-dr-eval-rev \
  --time=01:00:00 \
  --dependency="${TRAIN_DEPENDENCY}" \
  "${ROOT}/experiments/privileged_information_distillation/evaluate_student_sft.sh" \
  --adapter-dir "${DOT_ADAPTER}" \
  --adapter-dir "${LOSS_ADAPTER}" \
  --adapter-dir "${RANDOM_ADAPTER}" \
  --adapter-dir "${BASELINE}" \
  --split validation \
  --run-name validation_datarater_reverse_v1 \
  --max-new-tokens 1 \
  --continuous-margins \
  --continuous-margin-condition direct)"

printf 'score=%s\n' "${SCORE}"
printf 'random50=%s\n' "${RANDOM}"
printf 'loss50=%s\n' "${LOSS}"
printf 'dot50=%s\n' "${DOT}"
printf 'forward=%s\n' "${FORWARD}"
printf 'reverse=%s\n' "${REVERSE}"
