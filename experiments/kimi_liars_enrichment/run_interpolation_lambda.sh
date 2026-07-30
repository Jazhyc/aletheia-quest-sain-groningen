#!/bin/bash
# Evaluate cheap Phoenix 8/full-dose LoRA interpolations after training exits.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

wait_pid="${1:-}"
if [[ -n "${wait_pid}" ]]; then
  echo "waiting for pid=${wait_pid}" >&2
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 30
  done
fi

source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

anchor="results/blackbox/qwen9b_kimi_k3_openrouter_tvg_soft_full_r16_lr5e5_ep2_v1/adapter"
full="results/blackbox/qwen9b_kimi_k3_tvg_soft_full_plus_liars_r16_lr5e5_ep2_v1/adapter"
half="results/blackbox/qwen9b_kimi_k3_tvg_soft_full_plus_liars50_r16_lr5e5_ep2_v1/adapter"
interpolation_root="results/blackbox/kimi_k3_liars_enrichment_interpolation_v1"
arms=(
  "${anchor}"
  "${interpolation_root}/full25/adapter"
  "${interpolation_root}/full50/adapter"
  "${interpolation_root}/full75/adapter"
  "${full}"
  "${half}"
)
for adapter in "${arms[@]}"; do
  test -f "${adapter}/adapter_config.json"
  test -f "${adapter}/adapter_model.safetensors"
done

validation_args=(
  --split validation
  --run-name validation_interpolation_deadline
  --max-new-tokens 1
  --continuous-margins
  --continuous-margin-condition direct
)
for adapter in "${arms[@]}"; do
  validation_args+=(--adapter-dir "${adapter}")
done
python experiments/privileged_information_distillation/evaluate_student_sft.py \
  "${validation_args[@]}"

python experiments/kimi_liars_enrichment/evaluate_pilot_margins.py \
  --eval-artifact results/blackbox/liars_bench_pid_aug_v1/eval.jsonl \
  --output-dir "${interpolation_root}/liars_pilot_interpolation_deadline" \
  --adapter "anchor=${anchor}" \
  --adapter "full25=${interpolation_root}/full25/adapter" \
  --adapter "full50=${interpolation_root}/full50/adapter" \
  --adapter "full75=${interpolation_root}/full75/adapter" \
  --adapter "full=${full}" \
  --adapter "half=${half}"
