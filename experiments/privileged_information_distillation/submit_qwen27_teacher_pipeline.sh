#!/bin/bash
# Submit the frozen tiered Qwen-27B teacher-to-student workflow.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

METHOD="qwen9b_pid_varied_teacher_qwen27_adamw5e5_v1"
OUT="results/blackbox/${METHOD}"
SHARD0="${OUT}/teacher/shard_0/train.jsonl"
SHARD1="${OUT}/teacher/shard_1/train.jsonl"
STAGE1_AUDIT="${OUT}/teacher/audit_4096.json"
FINAL_AUDIT="${OUT}/teacher/audit_tiered.json"
MANIFEST4096="${OUT}/teacher/4096-valid-manifest.json"
MANIFEST8192="${OUT}/teacher/8192-valid-manifest.json"
MERGED="${OUT}/teacher/train.jsonl"
BASELINE="results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter"
CANDIDATE="${OUT}/adapter"

mkdir -p "${OUT}/teacher/shard_0" "${OUT}/teacher/shard_1"

submit_id() {
  local submitted
  submitted="$(sbatch --parsable "$@")"
  printf '%s' "${submitted%%;*}"
}

GEN0="$(submit_id \
  --job-name=aq-q27t-4k-s0 \
  --time=03:50:00 \
  experiments/privileged_information_distillation/run_teacher.sh \
  --config-name pid_teacher_qwen27_varied_v1 \
  "method=${METHOD}" \
  teacher.max_tokens=4096 \
  teacher.max_model_len=8192 \
  teacher.shard_count=2 \
  teacher.shard_index=0 \
  "teacher.artifact=${SHARD0}")"

GEN1="$(submit_id \
  --job-name=aq-q27t-4k-s1 \
  --time=03:50:00 \
  experiments/privileged_information_distillation/run_teacher.sh \
  --config-name pid_teacher_qwen27_varied_v1 \
  "method=${METHOD}" \
  teacher.max_tokens=4096 \
  teacher.max_model_len=8192 \
  teacher.shard_count=2 \
  teacher.shard_index=1 \
  "teacher.artifact=${SHARD1}")"

AUDIT4096="$(submit_id \
  --job-name=aq-q27t-audit4k \
  --dependency="afterok:${GEN0}:${GEN1}" \
  experiments/privileged_information_distillation/run_qwen_teacher_cache_audit.sh \
  --shard "${SHARD0}" \
  --shard "${SHARD1}" \
  --expected-total 2880 \
  --minimum-usable 2304 \
  --allow-unclosed \
  --allow-truncated-targets \
  --write-manifest "${MANIFEST4096}" \
  --output "${STAGE1_AUDIT}")"

RETRY0="$(submit_id \
  --job-name=aq-q27t-8k-s0 \
  --time=03:50:00 \
  --dependency="afterok:${AUDIT4096}" \
  experiments/privileged_information_distillation/run_teacher.sh \
  --config-name pid_teacher_qwen27_varied_v1 \
  "method=${METHOD}" \
  teacher.max_tokens=8192 \
  teacher.max_model_len=12288 \
  teacher.shard_count=2 \
  teacher.shard_index=0 \
  "teacher.artifact=${SHARD0}")"

RETRY1="$(submit_id \
  --job-name=aq-q27t-8k-s1 \
  --time=03:50:00 \
  --dependency="afterok:${AUDIT4096}" \
  experiments/privileged_information_distillation/run_teacher.sh \
  --config-name pid_teacher_qwen27_varied_v1 \
  "method=${METHOD}" \
  teacher.max_tokens=8192 \
  teacher.max_model_len=12288 \
  teacher.shard_count=2 \
  teacher.shard_index=1 \
  "teacher.artifact=${SHARD1}")"

AUDIT8192="$(submit_id \
  --job-name=aq-q27t-audit8k \
  --dependency="afterok:${RETRY0}:${RETRY1}" \
  experiments/privileged_information_distillation/run_qwen_teacher_cache_audit.sh \
  --shard "${SHARD0}" \
  --shard "${SHARD1}" \
  --expected-total 2880 \
  --minimum-usable 2794 \
  --allow-unclosed \
  --allow-truncated-targets \
  --verify-manifest "${MANIFEST4096}" \
  --write-manifest "${MANIFEST8192}" \
  --output "${OUT}/teacher/audit_8192.json")"

RETRY16_0="$(submit_id \
  --job-name=aq-q27t-16k-s0 \
  --time=03:50:00 \
  --dependency="afterok:${AUDIT8192}" \
  experiments/privileged_information_distillation/run_teacher.sh \
  --config-name pid_teacher_qwen27_varied_v1 \
  "method=${METHOD}" \
  teacher.max_tokens=16384 \
  teacher.max_model_len=20480 \
  teacher.shard_count=2 \
  teacher.shard_index=0 \
  "teacher.artifact=${SHARD0}")"

RETRY16_1="$(submit_id \
  --job-name=aq-q27t-16k-s1 \
  --time=03:50:00 \
  --dependency="afterok:${AUDIT8192}" \
  experiments/privileged_information_distillation/run_teacher.sh \
  --config-name pid_teacher_qwen27_varied_v1 \
  "method=${METHOD}" \
  teacher.max_tokens=16384 \
  teacher.max_model_len=20480 \
  teacher.shard_count=2 \
  teacher.shard_index=1 \
  "teacher.artifact=${SHARD1}")"

FINAL="$(submit_id \
  --job-name=aq-q27t-audit-final \
  --dependency="afterok:${RETRY16_0}:${RETRY16_1}" \
  experiments/privileged_information_distillation/run_qwen_teacher_cache_audit.sh \
  --shard "${SHARD0}" \
  --shard "${SHARD1}" \
  --expected-total 2880 \
  --minimum-usable 2877 \
  --maximum-label-imbalance 3 \
  --allow-unclosed \
  --allow-truncated-targets \
  --verify-manifest "${MANIFEST8192}" \
  --merged-output "${MERGED}" \
  --output "${FINAL_AUDIT}")"

STUDENT="$(submit_id \
  --job-name=aq-q27t-student \
  --dependency="afterok:${FINAL}" \
  experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_teacher_qwen27_varied_v1 \
  "method=${METHOD}" \
  "teacher.artifact=${MERGED}")"

EVAL="$(submit_id \
  --job-name=aq-q27t-eval \
  --dependency="afterok:${STUDENT}" \
  experiments/privileged_information_distillation/evaluate_student_sft.sh \
  --adapter-dir "${BASELINE}" \
  --adapter-dir "${CANDIDATE}" \
  --split validation \
  --run-name validation_qwen27_teacher_v1)"

printf 'generation_4096=%s,%s\n' "${GEN0}" "${GEN1}"
printf 'audit_4096=%s\n' "${AUDIT4096}"
printf 'selective_retry_8192=%s,%s\n' "${RETRY0}" "${RETRY1}"
printf 'audit_8192=%s\n' "${AUDIT8192}"
printf 'selective_retry_16384=%s,%s\n' "${RETRY16_0}" "${RETRY16_1}"
printf 'audit_merge=%s\n' "${FINAL}"
printf 'student=%s\n' "${STUDENT}"
printf 'validation=%s\n' "${EVAL}"
