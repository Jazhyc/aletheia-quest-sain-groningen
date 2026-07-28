#!/bin/bash
# Evaluate direct and post-reasoning margins for the clarified Qwen-397B student.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

ADAPTER="results/blackbox/qwen9b_qwen397_openrouter_explicit_tvg_binary_softonly_varied_v1/adapter"
PROMPT_CONFIG="configs/pid_qwen397_openrouter_explicit_post_reasoning_eval_v1.yaml"
RUN_NAME="validation_qwen397_openrouter_explicit_post_reasoning_v1"

JOB_ID="$(
  sbatch --parsable \
    --job-name=aq-q397-or-post \
    --time=00:20:00 \
    --gpus-per-node=rtx_pro_6000:1 \
    "${ROOT}/experiments/privileged_information_distillation/evaluate_student_sft.sh" \
    --adapter-dir "${ADAPTER}" \
    --split validation \
    --run-name "${RUN_NAME}" \
    --prompt-config "${PROMPT_CONFIG}" \
    --max-new-tokens 512 \
    --continuous-margins \
    --continuous-margin-condition direct \
    --continuous-margin-condition reasoning
)"

printf 'validation=%s\n' "${JOB_ID%%;*}"
