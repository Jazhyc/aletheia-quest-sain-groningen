#!/bin/bash
# Benchmark GRPO generation-batch sizing from the optimized Qwen-397B adapter.

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
export WANDB_DIR="${WANDB_DIR:-logs/wandb}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

KERNEL_PATH="${Q397_KERNEL_PATH:-/tmp/q397-fla}"
if [[ ! -d "${KERNEL_PATH}/fla/ops" ]]; then
  uv pip install \
    --python .venv/bin/python \
    --target "${KERNEL_PATH}" \
    flash-linear-attention==0.5.2 \
    fla-core==0.5.2 \
    --no-deps
fi
export PYTHONPATH="${KERNEL_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

BENCHMARK="q397soft_ep2_grpo_reason_speed_v1"
LOG_DIR="logs/lambda/${BENCHMARK}"
MAX_STEPS="${Q397_GRPO_BENCHMARK_STEPS:-16}"
mkdir -p "${LOG_DIR}" "${WANDB_DIR}"

run_condition() {
  local name="$1"
  local generation_batch_size="$2"
  local vllm_memory="$3"
  local method="${BENCHMARK}_${name}"
  local log="${LOG_DIR}/${name}.out"

  echo "condition=${name} generation_batch_size=${generation_batch_size} vllm_memory=${vllm_memory}"
  accelerate launch \
    --num_processes 1 \
    --main_process_port 0 \
    experiments/qwen_grpo_lora/run_qwen_grpo_lora.py \
    --config-name qwen_grpo_lora_q397_optimized_reasoning \
    "method=${method}" \
    "output_dir=results/blackbox/${method}" \
    "training.max_steps=${MAX_STEPS}" \
    "training.generation_batch_size=${generation_batch_size}" \
    "vllm.gpu_memory_utilization=${vllm_memory}" \
    "train_global_limit=128" \
    "validation_global_limit=8" \
    "wandb.enabled=false" \
    2>&1 | tee "${log}"
}

if ! run_condition gbs32_mem35 32 0.35; then
  echo "condition failed: gbs32_mem35" >&2
fi
if ! run_condition gbs64_mem50 64 0.50; then
  echo "condition failed: gbs64_mem50" >&2
fi
if ! run_condition gbs32_mem35_warm 32 0.35; then
  echo "condition failed: gbs32_mem35_warm" >&2
fi

python - "${LOG_DIR}" <<'PY'
import json
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
rows = []
for log_path in sorted(log_dir.glob("*.out")):
    text = log_path.read_text()
    runtime = re.findall(r"'train_runtime': ([0-9.]+)", text)
    steps_per_second = re.findall(r"'train_steps_per_second': ([0-9.]+)", text)
    rows.append({
        "condition": log_path.stem,
        "train_runtime": float(runtime[-1]) if runtime else None,
        "train_steps_per_second": float(steps_per_second[-1]) if steps_per_second else None,
    })
(log_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY
