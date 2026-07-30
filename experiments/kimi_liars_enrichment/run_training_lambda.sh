#!/bin/bash
# Train the full and half-dose Liars enrichment students, then compare on validation.

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

CAMPAIGN_METHOD="kimi_k3_liars_enrichment_scale_probe_v1"
LOG_DIR="logs/lambda/${CAMPAIGN_METHOD}"
RUN_NAME="validation_liars_scale_probe"
CACHE_ROOT="results/blackbox/kimi_k3_tvg_soft_full_plus_liars_v1/train"
ANCHOR_METHOD="qwen9b_kimi_k3_openrouter_tvg_soft_full_r16_lr5e5_ep2_v1"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

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

expected_student_sha="87cde0d4e6be1edb149a9cd68d4dcf2fbef8c8e8e28487d96569246b5a935fe7"
expected_soft_sha="2100781d05a2f61328568524233d2a6be583a95b10c28bb6b265e80abd64d273"
actual_student_sha="$(sha256sum "${CACHE_ROOT}/student_rows.jsonl" | cut -d' ' -f1)"
actual_soft_sha="$(sha256sum "${CACHE_ROOT}/soft_targets.jsonl" | cut -d' ' -f1)"
[[ "${actual_student_sha}" == "${expected_student_sha}" ]]
[[ "${actual_soft_sha}" == "${expected_soft_sha}" ]]
for artifact in soft_targets.jsonl student_rows.jsonl; do
  rows="$(wc -l < "${CACHE_ROOT}/${artifact}")"
  if [[ "${rows}" -ne 13149 ]]; then
    echo "expected 13149 rows in ${CACHE_ROOT}/${artifact}, found ${rows}" >&2
    exit 1
  fi
done

methods=(
  qwen9b_kimi_k3_tvg_soft_full_plus_liars_r16_lr5e5_ep2_v1
  qwen9b_kimi_k3_tvg_soft_full_plus_liars50_r16_lr5e5_ep2_v1
)
configs=(
  pid_kimi_k3_liars_binary_soft_full_r16_ep2_v1
  pid_kimi_k3_liars_binary_soft_half_r16_ep2_v1
)

for index in "${!methods[@]}"; do
  method="${methods[$index]}"
  config="${configs[$index]}"
  adapter="results/blackbox/${method}/adapter"
  migration="results/blackbox/${method}/peft_path_migration"

  if [[ -f "${adapter}/adapter_config.json" ]] \
    && [[ -f "${adapter}/adapter_model.safetensors" ]] \
    && [[ -f "${migration}/local_manifest.json" ]]; then
    echo "adapter already complete; skipping training: ${method}"
  else
    echo "training method=${method} config=${config}"
    python experiments/privileged_information_distillation/train_student_sft.py \
      --config-name "${config}" \
      "student.training.per_device_train_batch_size=${MICRO_BATCH}" \
      "student.training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION}" \
      "student.training.torch_compile=false"

    python experiments/heterogeneous_adapter_ensemble/migrate_qwen35_peft_paths.py \
      --local-only \
      --local-dir "${adapter}" \
      --work-dir "${migration}"
  fi
done

anchor_adapter="results/blackbox/${ANCHOR_METHOD}/adapter"
if [[ ! -f "${anchor_adapter}/adapter_model.safetensors" ]]; then
  cp submission/phoenix_wright_adapters/main/adapter_model.safetensors \
    "${anchor_adapter}/adapter_model.safetensors"
fi

eval_args=(
  --split validation
  --run-name "${RUN_NAME}"
  --max-new-tokens 1
  --continuous-margins
  --continuous-margin-condition direct
  --verify-lora-effect
  --adapter-dir "${anchor_adapter}"
)
for method in "${methods[@]}"; do
  eval_args+=(--adapter-dir "results/blackbox/${method}/adapter")
done

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  "${eval_args[@]}"

for method in "${ANCHOR_METHOD}" "${methods[@]}"; do
  sha256sum "results/blackbox/${method}/adapter/adapter_model.safetensors"
done
