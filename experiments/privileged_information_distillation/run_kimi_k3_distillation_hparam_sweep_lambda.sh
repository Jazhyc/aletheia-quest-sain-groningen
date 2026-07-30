#!/bin/bash
# Test whether the Q397 binary-soft optimum transfers to the Kimi K3 teacher.

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

CAMPAIGN_METHOD="kimi_k3_soft_distillation_hparam_sweep_v1"
LOG_DIR="logs/lambda/${CAMPAIGN_METHOD}"
RUN_NAME="validation_hparam_sweep"
CACHE_ROOT="results/blackbox/kimi_k3_fireworks_nothink_tvg_binary_logit_v1/train"
BASE_CONFIG="pid_kimi_k3_openrouter_tvg_binary_soft_distillation_v1"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/sweep-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

python - <<'PY'
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.utils.import_utils import is_flash_linear_attention_available

if not is_flash_linear_attention_available():
    raise RuntimeError("Transformers did not recognize Flash Linear Attention")
print(f"qwen_fast_kernel={chunk_gated_delta_rule.__module__}", flush=True)
PY

sha256sum -c "${CACHE_ROOT}/SHA256SUMS"
for artifact in generations.jsonl soft_targets.jsonl student_rows.jsonl; do
  rows="$(wc -l < "${CACHE_ROOT}/${artifact}")"
  if [[ "${rows}" -ne 2880 ]]; then
    echo "expected 2880 rows in ${CACHE_ROOT}/${artifact}, found ${rows}" >&2
    exit 1
  fi
done

learning_rates=(1e-5 2e-5 5e-5 1e-4)
learning_rate_names=(1e5 2e5 5e5 1e4)
epochs=(0.5 1.0 2.0)
epoch_names=(ep05 ep1 ep2)

method_name() {
  printf 'qwen9b_kimi_k3_openrouter_tvg_soft_r16_lr%s_%s_v1' "$1" "$2"
}

methods=()
for lr_index in "${!learning_rates[@]}"; do
  for epoch_index in "${!epochs[@]}"; do
    learning_rate="${learning_rates[$lr_index]}"
    learning_rate_name="${learning_rate_names[$lr_index]}"
    epoch="${epochs[$epoch_index]}"
    epoch_name="${epoch_names[$epoch_index]}"
    method="$(method_name "${learning_rate_name}" "${epoch_name}")"
    adapter="results/blackbox/${method}/adapter"
    migration="results/blackbox/${method}/peft_path_migration"

    if [[ -f "${adapter}/adapter_config.json" ]] \
      && [[ -f "${migration}/local_manifest.json" ]]; then
      echo "adapter already complete; skipping training: ${method}"
    else
      echo "training method=${method} lr=${learning_rate} epochs=${epoch}"
      python experiments/privileged_information_distillation/train_student_sft.py \
        --config-name "${BASE_CONFIG}" \
        "method=${method}" \
        "output_dir=results/blackbox/${method}" \
        "student.output_dir=results/blackbox/${method}/adapter" \
        "student.training.learning_rate=${learning_rate}" \
        "student.training.num_train_epochs=${epoch}" \
        "student.training.per_device_train_batch_size=${MICRO_BATCH}" \
        "student.training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION}" \
        "student.training.torch_compile=false"

      python experiments/heterogeneous_adapter_ensemble/migrate_qwen35_peft_paths.py \
        --local-only \
        --local-dir "${adapter}" \
        --work-dir "${migration}"
    fi
    methods+=("${method}")
  done
done

eval_args=(
  --split validation
  --run-name "${RUN_NAME}"
  --max-new-tokens 1
  --continuous-margins
  --continuous-margin-condition direct
  --verify-lora-effect
)
for method in "${methods[@]}"; do
  eval_args+=(--adapter-dir "results/blackbox/${method}/adapter")
done

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  "${eval_args[@]}"
python experiments/privileged_information_distillation/summarize_kimi_k3_distillation_sweep.py \
  --run-name "${RUN_NAME}" \
  --output-dir "results/blackbox/${CAMPAIGN_METHOD}/rank16" \
  "${methods[@]/#/--method=}"
