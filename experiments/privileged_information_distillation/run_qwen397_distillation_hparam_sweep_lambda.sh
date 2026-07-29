#!/bin/bash
# Optimize the pure Qwen-397B binary-soft student on a persistent Lambda H100.

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

CAMPAIGN_METHOD="qwen397_soft_distillation_hparam_sweep_v1"
LOG_DIR="logs/lambda/${CAMPAIGN_METHOD}"
RUN_NAME="validation_hparam_sweep"
CACHE_ROOT="results/blackbox/qwen35_397b_fp8_nothink_truth_value_binary_logit_v1/train"
BASE_CONFIG="pid_qwen397_tvg_binary_soft_distillation_v1"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/sweep-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

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
  local rank="$1"
  local learning_rate_name="$2"
  local epoch_name="$3"
  printf 'qwen9b_qwen397_tvg_soft_r%s_lr%s_%s_v1' \
    "${rank}" "${learning_rate_name}" "${epoch_name}"
}

train_adapter() {
  local rank="$1"
  local alpha="$2"
  local learning_rate="$3"
  local learning_rate_name="$4"
  local epoch="$5"
  local epoch_name="$6"
  local method
  local adapter
  local migration
  method="$(method_name "${rank}" "${learning_rate_name}" "${epoch_name}")"
  adapter="results/blackbox/${method}/adapter"
  migration="results/blackbox/${method}/peft_path_migration"

  if [[ -f "${adapter}/adapter_config.json" ]] \
    && [[ -f "${migration}/local_manifest.json" ]]; then
    echo "adapter already complete; skipping training: ${method}"
    return
  fi

  echo "training method=${method} rank=${rank} alpha=${alpha} lr=${learning_rate} epochs=${epoch}"
  python experiments/privileged_information_distillation/train_student_sft.py \
    --config-name "${BASE_CONFIG}" \
    "method=${method}" \
    "output_dir=results/blackbox/${method}" \
    "student.output_dir=results/blackbox/${method}/adapter" \
    "student.lora.r=${rank}" \
    "student.lora.alpha=${alpha}" \
    "student.training.learning_rate=${learning_rate}" \
    "student.training.num_train_epochs=${epoch}" \
    "student.training.per_device_train_batch_size=${MICRO_BATCH}" \
    "student.training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION}" \
    "student.training.torch_compile=false"

  python experiments/heterogeneous_adapter_ensemble/migrate_qwen35_peft_paths.py \
    --local-only \
    --local-dir "${adapter}" \
    --work-dir "${migration}"
}

evaluate_methods() {
  local rank="$1"
  shift
  local methods=("$@")
  local eval_args=(
    --split validation
    --run-name "${RUN_NAME}"
    --max-new-tokens 1
    --continuous-margins
    --continuous-margin-condition direct
    --verify-lora-effect
  )
  local method
  for method in "${methods[@]}"; do
    eval_args+=(--adapter-dir "results/blackbox/${method}/adapter")
  done

  echo "evaluating rank=${rank} adapters=${#methods[@]}"
  python experiments/privileged_information_distillation/evaluate_student_sft.py \
    "${eval_args[@]}"
  python experiments/privileged_information_distillation/summarize_qwen397_distillation_sweep.py \
    --run-name "${RUN_NAME}" \
    --output-dir "results/blackbox/${CAMPAIGN_METHOD}/rank${rank}" \
    "${methods[@]/#/--method=}"
}

rank16_methods=()
for lr_index in "${!learning_rates[@]}"; do
  for epoch_index in "${!epochs[@]}"; do
    train_adapter \
      16 32 \
      "${learning_rates[$lr_index]}" "${learning_rate_names[$lr_index]}" \
      "${epochs[$epoch_index]}" "${epoch_names[$epoch_index]}"
    rank16_methods+=("$(
      method_name 16 "${learning_rate_names[$lr_index]}" "${epoch_names[$epoch_index]}"
    )")
  done
done
evaluate_methods 16 "${rank16_methods[@]}"

read -r best_lr best_lr_name best_epoch best_epoch_name < <(
  python experiments/privileged_information_distillation/summarize_qwen397_distillation_sweep.py \
    --run-name "${RUN_NAME}" \
    --best-shell \
    "${rank16_methods[@]/#/--method=}"
)
echo "rank16 winner lr=${best_lr} epochs=${best_epoch}"

# Rank 24 gets a small response-surface follow-up: all epoch lengths at the
# rank-16-winning LR, plus every LR at the rank-16-winning epoch.
rank24_methods=()
for epoch_index in "${!epochs[@]}"; do
  train_adapter \
    24 48 \
    "${best_lr}" "${best_lr_name}" \
    "${epochs[$epoch_index]}" "${epoch_names[$epoch_index]}"
  rank24_methods+=("$(
    method_name 24 "${best_lr_name}" "${epoch_names[$epoch_index]}"
  )")
done
for lr_index in "${!learning_rates[@]}"; do
  method="$(method_name 24 "${learning_rate_names[$lr_index]}" "${best_epoch_name}")"
  if [[ " ${rank24_methods[*]} " == *" ${method} "* ]]; then
    continue
  fi
  train_adapter \
    24 48 \
    "${learning_rates[$lr_index]}" "${learning_rate_names[$lr_index]}" \
    "${best_epoch}" "${best_epoch_name}"
  rank24_methods+=("${method}")
done
evaluate_methods 24 "${rank24_methods[@]}"

python experiments/privileged_information_distillation/summarize_qwen397_distillation_sweep.py \
  --run-name "${RUN_NAME}" \
  --output-dir "results/blackbox/${CAMPAIGN_METHOD}/all" \
  "${rank16_methods[@]/#/--method=}" \
  "${rank24_methods[@]/#/--method=}"
