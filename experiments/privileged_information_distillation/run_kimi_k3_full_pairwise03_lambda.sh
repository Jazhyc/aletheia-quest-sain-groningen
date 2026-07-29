#!/bin/bash
# Train and jointly validate the frozen full-data Kimi BCE-plus-pairwise student.

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

METHOD="qwen9b_kimi_k3_openrouter_tvg_soft_pairwise03_full_r16_lr5e5_ep2_v1"
BASE_CONFIG="pid_kimi_k3_openrouter_tvg_binary_soft_pairwise03_full_r16_ep2_v1"
ANCHOR_METHOD="qwen9b_kimi_k3_openrouter_tvg_soft_full_r16_lr5e5_ep2_v1"
CACHE_ROOT="results/blackbox/kimi_k3_fireworks_nothink_tvg_binary_logit_full_v1/train"
ADAPTER="results/blackbox/${METHOD}/adapter"
ANCHOR_ADAPTER="results/blackbox/${ANCHOR_METHOD}/adapter"
MIGRATION="results/blackbox/${METHOD}/peft_path_migration"
RUN_NAME="validation_direct_margin_pairwise_comparison"
LOG_DIR="logs/lambda/${METHOD}"
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
if [[ ! -f "${ANCHOR_ADAPTER}/adapter_config.json" ]]; then
  echo "required AdamW anchor adapter is missing: ${ANCHOR_ADAPTER}" >&2
  exit 1
fi

if [[ ! -f "${ADAPTER}/adapter_config.json" ]] \
  || [[ ! -f "${MIGRATION}/local_manifest.json" ]]; then
  python experiments/privileged_information_distillation/train_student_sft.py \
    --config-name "${BASE_CONFIG}" \
    "student.training.per_device_train_batch_size=${MICRO_BATCH}" \
    "student.training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION}" \
    "student.training.torch_compile=false"

  python experiments/heterogeneous_adapter_ensemble/migrate_qwen35_peft_paths.py \
    --local-only \
    --local-dir "${ADAPTER}" \
    --work-dir "${MIGRATION}"
else
  echo "adapter already complete; skipping training: ${METHOD}"
fi

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  --adapter-dir "${ANCHOR_ADAPTER}" \
  --adapter-dir "${ADAPTER}" \
  --split validation \
  --run-name "${RUN_NAME}" \
  --max-new-tokens 1 \
  --continuous-margins \
  --continuous-margin-condition direct \
  --verify-lora-effect

sha256sum "${ADAPTER}/adapter_model.safetensors"
