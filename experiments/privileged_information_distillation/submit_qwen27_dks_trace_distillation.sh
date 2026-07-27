#!/bin/bash
# Submit the frozen full-trace Qwen-27B D/K/S distillation workflow.

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

GENERATION_METHOD="qwen27b_reason_ensemble_dks_member4096_softtrain_v1"
CACHE_METHOD="qwen27b_dks_full_trace_soft_teacher_v1"
STUDENT_METHOD="qwen9b_pid_qwen27_dks_fulltrace_soft90_v1"
SMOKE_METHOD="${STUDENT_METHOD}_smoke"
EXPORT_WORKTREE="ALL,QWEN27_DKS_WORKTREE=${WORKTREE_ROOT}"

GENERATION="$(submit_id \
  --job-name=aq-q27-dks-traces \
  --time=03:50:00 \
  "${PROJECT_ROOT}/experiments/blackbox/run_judge.sh" \
  --config-path "${WORKTREE_ROOT}/configs/judge_ensemble" \
  --config-name blackbox_reasoning_qwen27b_ensemble_dks_member4096_v1 \
  "method=${GENERATION_METHOD}" \
  split=train \
  +dataset_name_contains=varied-deception \
  judge.max_prompt_chars=3000)"

CACHE="$(submit_id \
  --job-name=aq-q27-dks-cache \
  --dependency="afterok:${GENERATION}" \
  --export="${EXPORT_WORKTREE}" \
  "${WORKTREE_ROOT}/experiments/privileged_information_distillation/run_qwen27_dks_trace_target_build.sh")"

SMOKE="$(submit_id \
  --job-name=aq-q27-dks-smoke \
  --time=01:00:00 \
  --dependency="afterok:${CACHE}" \
  --export="${EXPORT_WORKTREE}" \
  "${WORKTREE_ROOT}/experiments/privileged_information_distillation/run_qwen27_dks_trace_student.sh" \
  "method=${SMOKE_METHOD}" \
  "output_dir=results/blackbox/${SMOKE_METHOD}" \
  student.train_limit=96)"

STUDENT="$(submit_id \
  --job-name=aq-q27-dks-student \
  --dependency="afterok:${SMOKE}" \
  --export="${EXPORT_WORKTREE}" \
  "${WORKTREE_ROOT}/experiments/privileged_information_distillation/run_qwen27_dks_trace_student.sh" \
  "method=${STUDENT_METHOD}" \
  "output_dir=results/blackbox/${STUDENT_METHOD}")"

EVAL="$(submit_id \
  --job-name=aq-q27-dks-eval \
  --dependency="afterok:${STUDENT}" \
  --export="${EXPORT_WORKTREE}" \
  "${WORKTREE_ROOT}/experiments/privileged_information_distillation/run_qwen27_dks_trace_eval.sh" \
  --adapter-dir "results/blackbox/${STUDENT_METHOD}/adapter" \
  --split train \
  --splits-dir "results/blackbox/${CACHE_METHOD}/holdout_splits" \
  --run-name holdout_dks_fulltrace_v1 \
  --max-new-tokens 4096 \
  --max-model-len 8192 \
  --prompt-condition "details=${WORKTREE_ROOT}/configs/pid_qwen27_dks_eval_details.yaml" \
  --prompt-condition "known=${WORKTREE_ROOT}/configs/pid_qwen27_dks_eval_known.yaml" \
  --prompt-condition "scrutiny=${WORKTREE_ROOT}/configs/pid_qwen27_dks_eval_scrutiny.yaml" \
  --continuous-rating-margins \
  --continuous-margin-condition direct \
  --continuous-margin-condition reasoning \
  --aggregate-max)"

printf 'qwen27_trace_generation=%s\n' "${GENERATION}"
printf 'trace_cache=%s\n' "${CACHE}"
printf 'student_smoke=%s\n' "${SMOKE}"
printf 'student_full=%s\n' "${STUDENT}"
printf 'holdout_eval=%s\n' "${EVAL}"
