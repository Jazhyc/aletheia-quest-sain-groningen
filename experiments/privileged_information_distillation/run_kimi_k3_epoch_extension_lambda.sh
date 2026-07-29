#!/bin/bash
# Extend the validation-winning Kimi K3 LR to fresh three/four-epoch runs.

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
export PYTHONPATH="${KERNEL_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

CAMPAIGN_METHOD="kimi_k3_soft_distillation_epoch_extension_v1"
UPSTREAM_METHOD="kimi_k3_soft_distillation_hparam_sweep_v1"
UPSTREAM_PID_FILE="logs/lambda/${UPSTREAM_METHOD}/sweep.pid"
UPSTREAM_SUMMARY="results/blackbox/${UPSTREAM_METHOD}/rank16/summary.json"
LOG_DIR="logs/lambda/${CAMPAIGN_METHOD}"
RUN_NAME="validation_epoch_extension"
BASE_CONFIG="pid_kimi_k3_openrouter_tvg_binary_soft_distillation_v1"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/extension-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

if [[ ! -f "${UPSTREAM_PID_FILE}" ]]; then
  echo "missing upstream PID file: ${UPSTREAM_PID_FILE}" >&2
  exit 1
fi
upstream_pid="$(cat "${UPSTREAM_PID_FILE}")"
while kill -0 "${upstream_pid}" 2>/dev/null; do
  echo "waiting_for_sweep_pid=${upstream_pid}"
  sleep 30
done
if [[ ! -f "${UPSTREAM_SUMMARY}" ]]; then
  echo "upstream sweep exited without summary: ${UPSTREAM_SUMMARY}" >&2
  exit 1
fi

read -r winning_lr winning_lr_name winning_epoch_name < <(
  python - "${UPSTREAM_SUMMARY}" <<'PY'
import json
import sys

best = json.load(open(sys.argv[1], encoding="utf-8"))[0]
print(best["lr"], best["lr_name"], best["epoch_name"])
PY
)
winner="qwen9b_kimi_k3_openrouter_tvg_soft_r16_lr${winning_lr_name}_${winning_epoch_name}_v1"
echo "upstream_winner=${winner} lr=${winning_lr}"

methods=("${winner}")
for epoch in 3.0 4.0; do
  epoch_name="ep${epoch%%.*}"
  method="qwen9b_kimi_k3_openrouter_tvg_soft_r16_lr${winning_lr_name}_${epoch_name}_v1"
  adapter="results/blackbox/${method}/adapter"
  migration="results/blackbox/${method}/peft_path_migration"
  if [[ -f "${adapter}/adapter_config.json" ]] \
    && [[ -f "${migration}/local_manifest.json" ]]; then
    echo "adapter already complete; skipping training: ${method}"
  else
    echo "training method=${method} lr=${winning_lr} epochs=${epoch}"
    python experiments/privileged_information_distillation/train_student_sft.py \
      --config-name "${BASE_CONFIG}" \
      "method=${method}" \
      "output_dir=results/blackbox/${method}" \
      "student.output_dir=results/blackbox/${method}/adapter" \
      "student.training.learning_rate=${winning_lr}" \
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
  --output-dir "results/blackbox/${CAMPAIGN_METHOD}" \
  "${methods[@]/#/--method=}"
