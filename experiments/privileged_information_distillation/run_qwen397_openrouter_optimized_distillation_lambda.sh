#!/bin/bash
# Train the clarified-prompt Qwen-397B soft student with the optimized H100 recipe.

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

KERNEL_PATH="${Q397_KERNEL_PATH:-/tmp/q397-fla}"
MICRO_BATCH="${Q397_MICRO_BATCH:-8}"
GRADIENT_ACCUMULATION="${Q397_GRADIENT_ACCUMULATION:-4}"
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

METHOD="qwen9b_qwen397_openrouter_explicit_tvg_soft_r16_lr5e5_ep2_v1"
BASE_CONFIG="pid_qwen397_openrouter_explicit_tvg_binary_soft_distillation_v1"
CACHE_ROOT="results/blackbox/qwen35_397b_openrouter_nothink_tvg_binary_logit_explicit_digits_v1/train"
ADAPTER="results/blackbox/${METHOD}/adapter"
MIGRATION="results/blackbox/${METHOD}/peft_path_migration"
RUN_NAME="validation_optimized_recipe"
LOG_DIR="logs/lambda/${METHOD}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

sha256sum -c "${CACHE_ROOT}/SHA256SUMS"
for artifact in generations.jsonl soft_targets.jsonl student_rows.jsonl; do
  rows="$(wc -l < "${CACHE_ROOT}/${artifact}")"
  if [[ "${rows}" -ne 2880 ]]; then
    echo "expected 2880 rows in ${CACHE_ROOT}/${artifact}, found ${rows}" >&2
    exit 1
  fi
done

if [[ ! -f "${ADAPTER}/adapter_config.json" ]] \
  || [[ ! -f "${MIGRATION}/local_manifest.json" ]]; then
  python experiments/privileged_information_distillation/train_student_sft.py \
    --config-name "${BASE_CONFIG}" \
    "method=${METHOD}" \
    "output_dir=results/blackbox/${METHOD}" \
    "student.output_dir=${ADAPTER}" \
    "student.lora.r=16" \
    "student.lora.alpha=32" \
    "student.training.learning_rate=5e-5" \
    "student.training.num_train_epochs=2.0" \
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
  --adapter-dir "${ADAPTER}" \
  --split validation \
  --run-name "${RUN_NAME}" \
  --max-new-tokens 1 \
  --continuous-margins \
  --continuous-margin-condition direct \
  --verify-lora-effect

sha256sum "${ADAPTER}/adapter_model.safetensors"
