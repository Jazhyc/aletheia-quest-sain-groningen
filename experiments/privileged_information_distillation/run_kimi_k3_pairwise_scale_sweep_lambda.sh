#!/bin/bash
# Select a grouped-pair loss scale internally, then retrain and validate once.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ -f "${HOME}/.config/aletheia/runtime.env" ]]; then
  source "${HOME}/.config/aletheia/runtime.env"
fi
if [[ -f "${HOME}/.config/aletheia/secrets.env" ]]; then
  source "${HOME}/.config/aletheia/secrets.env"
fi
source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

KERNEL_PATH="${KIMI_KERNEL_PATH:-/tmp/kimi-fla}"
MICRO_BATCH="${KIMI_MICRO_BATCH:-8}"
GRADIENT_ACCUMULATION="${KIMI_GRADIENT_ACCUMULATION:-4}"
if [[ ! -d "${KERNEL_PATH}/fla/ops" ]]; then
  uv pip install \
    --python .venv/bin/python \
    --target "${KERNEL_PATH}" \
    flash-linear-attention==0.5.2 \
    fla-core==0.5.2 \
    --no-deps
fi
export PYTHONPATH="${KERNEL_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
python - <<'PY'
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from transformers.utils.import_utils import is_flash_linear_attention_available

if not is_flash_linear_attention_available():
    raise RuntimeError("Transformers did not recognize Flash Linear Attention")
print(
    "qwen_fast_kernel="
    f"{chunk_gated_delta_rule.__module__} "
    f"recurrent_kernel={fused_recurrent_gated_delta_rule.__module__}",
    flush=True,
)
PY

BASE_CONFIG="pid_kimi_k3_openrouter_tvg_binary_soft_pairwise03_full_r16_ep2_v1"
CACHE_ROOT="results/blackbox/kimi_k3_fireworks_nothink_tvg_binary_logit_full_v1/train"
SWEEP_ROOT="results/blackbox/kimi_k3_pairwise_scale_grouped_sweep_v1"
TRAIN_MANIFEST="${SWEEP_ROOT}/train_manifest.jsonl"
HOLDOUT_SPLITS="${SWEEP_ROOT}/holdout_splits"
SELECTION_JSON="${SWEEP_ROOT}/selection.json"
ANCHOR_ADAPTER="results/blackbox/qwen9b_kimi_k3_openrouter_tvg_soft_full_r16_lr5e5_ep2_v1/adapter"
OLD_PAIRWISE_ADAPTER="results/blackbox/qwen9b_kimi_k3_openrouter_tvg_soft_pairwise03_full_r16_lr5e5_ep2_v1/adapter"
LOG_DIR="logs/lambda/kimi_k3_pairwise_scale_grouped_sweep_v1"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

sha256sum -c "${CACHE_ROOT}/SHA256SUMS"
for artifact in soft_targets.jsonl student_rows.jsonl; do
  rows="$(wc -l < "${CACHE_ROOT}/${artifact}")"
  if [[ "${rows}" -ne 6573 ]]; then
    echo "expected 6573 rows in ${CACHE_ROOT}/${artifact}, found ${rows}" >&2
    exit 1
  fi
done
for required_adapter in "${ANCHOR_ADAPTER}" "${OLD_PAIRWISE_ADAPTER}"; do
  if [[ ! -f "${required_adapter}/adapter_config.json" ]]; then
    echo "required comparison adapter is missing: ${required_adapter}" >&2
    exit 1
  fi
done

if [[ ! -f "${SWEEP_ROOT}/audit.json" ]]; then
  python \
    experiments/privileged_information_distillation/build_kimi_pairwise_holdout.py \
    --student-rows "${CACHE_ROOT}/student_rows.jsonl" \
    --output-dir "${SWEEP_ROOT}" \
    --train-fraction 0.8 \
    --seed 20260729 \
    --expected-rows 6573
fi

WEIGHTS=(0.0 0.1 0.3 1.0)
TAGS=(00 01 03 10)
HOLDOUT_ADAPTERS=()
for index in "${!WEIGHTS[@]}"; do
  weight="${WEIGHTS[$index]}"
  tag="${TAGS[$index]}"
  method="qwen9b_kimi_k3_openrouter_tvg_soft_pairwise_scale${tag}_grouped_holdout80_r16_ep2_v1"
  output="results/blackbox/${method}"
  adapter="${output}/adapter"
  migration="${output}/peft_path_migration"
  HOLDOUT_ADAPTERS+=("${adapter}")
  if [[ -f "${adapter}/adapter_config.json" ]] \
    && [[ -f "${migration}/local_manifest.json" ]]; then
    echo "holdout adapter already complete; skipping training: ${method}"
    continue
  fi
  python experiments/privileged_information_distillation/train_student_sft.py \
    --config-name "${BASE_CONFIG}" \
    "method=${method}" \
    "output_dir=${output}" \
    "student.output_dir=${adapter}" \
    "student.selection_manifest=${TRAIN_MANIFEST}" \
    "student.training.pairwise_loss_weight=${weight}" \
    "student.training.paired_batching=true" \
    "student.training.paired_batching_mode=same_dataset" \
    "student.training.per_device_train_batch_size=${MICRO_BATCH}" \
    "student.training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION}" \
    "student.training.torch_compile=false"

  python experiments/heterogeneous_adapter_ensemble/migrate_qwen35_peft_paths.py \
    --local-only \
    --local-dir "${adapter}" \
    --work-dir "${migration}"
done

FORWARD_ARGS=()
for adapter in "${HOLDOUT_ADAPTERS[@]}"; do
  FORWARD_ARGS+=(--adapter-dir "${adapter}")
done
python experiments/privileged_information_distillation/evaluate_student_sft.py \
  "${FORWARD_ARGS[@]}" \
  --split train \
  --splits-dir "${HOLDOUT_SPLITS}" \
  --run-name internal_holdout_forward \
  --max-new-tokens 1 \
  --continuous-margins \
  --continuous-margin-condition direct

REVERSE_ARGS=()
for ((index=${#HOLDOUT_ADAPTERS[@]}-1; index>=0; index--)); do
  REVERSE_ARGS+=(--adapter-dir "${HOLDOUT_ADAPTERS[$index]}")
done
python experiments/privileged_information_distillation/evaluate_student_sft.py \
  "${REVERSE_ARGS[@]}" \
  --split train \
  --splits-dir "${HOLDOUT_SPLITS}" \
  --run-name internal_holdout_reverse \
  --max-new-tokens 1 \
  --continuous-margins \
  --continuous-margin-condition direct

SELECT_ARGS=()
for index in "${!WEIGHTS[@]}"; do
  SELECT_ARGS+=(
    --candidate
    "${WEIGHTS[$index]}=${HOLDOUT_ADAPTERS[$index]}"
  )
done
python \
  experiments/privileged_information_distillation/select_kimi_pairwise_scale.py \
  "${SELECT_ARGS[@]}" \
  --run-name internal_holdout_forward \
  --run-name internal_holdout_reverse \
  --tie-tolerance 0.001 \
  --output "${SELECTION_JSON}"

SELECTED_WEIGHT="$(
  python - "${SELECTION_JSON}" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["selected_weight"])
PY
)"
case "${SELECTED_WEIGHT}" in
  0.0) SELECTED_TAG="00" ;;
  0.1) SELECTED_TAG="01" ;;
  0.3) SELECTED_TAG="03" ;;
  1.0) SELECTED_TAG="10" ;;
  *)
    echo "unexpected selected pairwise weight: ${SELECTED_WEIGHT}" >&2
    exit 1
    ;;
esac

FINAL_METHOD="qwen9b_kimi_k3_openrouter_tvg_soft_pairwise_scale${SELECTED_TAG}_grouped_full_r16_lr5e5_ep2_v1"
FINAL_OUTPUT="results/blackbox/${FINAL_METHOD}"
FINAL_ADAPTER="${FINAL_OUTPUT}/adapter"
FINAL_MIGRATION="${FINAL_OUTPUT}/peft_path_migration"
if [[ ! -f "${FINAL_ADAPTER}/adapter_config.json" ]] \
  || [[ ! -f "${FINAL_MIGRATION}/local_manifest.json" ]]; then
  python experiments/privileged_information_distillation/train_student_sft.py \
    --config-name "${BASE_CONFIG}" \
    "method=${FINAL_METHOD}" \
    "output_dir=${FINAL_OUTPUT}" \
    "student.output_dir=${FINAL_ADAPTER}" \
    "student.training.pairwise_loss_weight=${SELECTED_WEIGHT}" \
    "student.training.paired_batching=true" \
    "student.training.paired_batching_mode=same_dataset" \
    "student.training.per_device_train_batch_size=${MICRO_BATCH}" \
    "student.training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION}" \
    "student.training.torch_compile=false"

  python experiments/heterogeneous_adapter_ensemble/migrate_qwen35_peft_paths.py \
    --local-only \
    --local-dir "${FINAL_ADAPTER}" \
    --work-dir "${FINAL_MIGRATION}"
else
  echo "selected full-data adapter already complete: ${FINAL_METHOD}"
fi

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  --adapter-dir "${ANCHOR_ADAPTER}" \
  --adapter-dir "${OLD_PAIRWISE_ADAPTER}" \
  --adapter-dir "${FINAL_ADAPTER}" \
  --split validation \
  --run-name validation_grouped_scale_forward \
  --max-new-tokens 1 \
  --continuous-margins \
  --continuous-margin-condition direct \
  --verify-lora-effect

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  --adapter-dir "${FINAL_ADAPTER}" \
  --adapter-dir "${OLD_PAIRWISE_ADAPTER}" \
  --adapter-dir "${ANCHOR_ADAPTER}" \
  --split validation \
  --run-name validation_grouped_scale_reverse \
  --max-new-tokens 1 \
  --continuous-margins \
  --continuous-margin-condition direct

sha256sum "${FINAL_ADAPTER}/adapter_model.safetensors"
echo "selected_weight=${SELECTED_WEIGHT}"
echo "selected_method=${FINAL_METHOD}"
