#!/bin/bash
# Submit the Qwen3.5-27B binary Truth Value Guard soft-distillation workflow.

set -euo pipefail

PROJECT_ROOT="$(pwd)"
WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  echo "run this from the primary project root containing .venv" >&2
  exit 1
fi

submit_id() {
  local submitted
  submitted="$(sbatch --parsable "$@")"
  printf '%s' "${submitted%%;*}"
}

TEACHER_METHOD="qwen35_27b_nothink_truth_value_binary_logit_v1"
STUDENT_METHOD="qwen9b_pid_qwen27_tvg_binary_soft_varied_v1"
BASELINE="results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter"
CANDIDATE="results/blackbox/${STUDENT_METHOD}/adapter"
EXPORT_WORKTREE="ALL,QWEN27_TVG_WORKTREE=${WORKTREE_ROOT}"

TEACHER="$(submit_id \
  --job-name=aq-q27-tvg-train \
  --time=01:00:00 \
  "${PROJECT_ROOT}/experiments/blackbox/run_judge.sh" \
  --config-path "${WORKTREE_ROOT}/configs/single_judges" \
  --config-name blackbox_reasoning_nothink_truth_value_binary_logit_qwen35_27b_v1 \
  "method=${TEACHER_METHOD}" \
  split=train \
  +dataset_name_contains=varied-deception)"

CACHE="$(submit_id \
  --job-name=aq-q27-tvg-cache \
  --dependency="afterok:${TEACHER}" \
  --export="${EXPORT_WORKTREE}" \
  "${WORKTREE_ROOT}/experiments/privileged_information_distillation/run_qwen27_tvg_soft_target_build.sh")"

STUDENT="$(submit_id \
  --job-name=aq-q27-tvg-student \
  --dependency="afterok:${CACHE}" \
  "${PROJECT_ROOT}/experiments/privileged_information_distillation/run_student_sft.sh" \
  --config-path "${WORKTREE_ROOT}/configs" \
  --config-name pid_qwen27_tvg_binary_soft_distillation_v1)"

EVAL="$(submit_id \
  --job-name=aq-q27-tvg-eval \
  --dependency="afterok:${STUDENT}" \
  "${PROJECT_ROOT}/experiments/privileged_information_distillation/evaluate_student_sft.sh" \
  --adapter-dir "${BASELINE}" \
  --adapter-dir "${CANDIDATE}" \
  --split validation \
  --run-name validation_qwen27_tvg_binary_soft_v1 \
  --max-new-tokens 1 \
  --continuous-margins \
  --continuous-margin-condition direct)"

printf 'teacher_cache=%s\n' "${TEACHER}"
printf 'soft_target_build=%s\n' "${CACHE}"
printf 'student=%s\n' "${STUDENT}"
printf 'validation=%s\n' "${EVAL}"
