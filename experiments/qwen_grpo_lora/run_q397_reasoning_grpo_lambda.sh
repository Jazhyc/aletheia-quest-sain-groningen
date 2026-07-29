#!/bin/bash
# Continue the optimized Qwen-397B distilled adapter with reasoning GRPO.

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
export PYTHONPATH="${KERNEL_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

METHOD="${Q397_GRPO_METHOD:-qwen9b_q397soft_ep2_grpo_reason_r16_v1}"
GENERATION_BATCH_SIZE="${Q397_GRPO_GENERATION_BATCH_SIZE:-32}"
VLLM_MEMORY="${Q397_GRPO_VLLM_MEMORY:-0.35}"
MUON_LR="${Q397_GRPO_MUON_LR:-3e-5}"
ADAMW_LR="${Q397_GRPO_ADAMW_LR:-1e-6}"
EPOCHS="${Q397_GRPO_EPOCHS:-1.0}"
TEMPERATURE="${Q397_GRPO_TEMPERATURE:-1.2}"
MAX_COMPLETION_LENGTH="${Q397_GRPO_MAX_COMPLETION_LENGTH:-256}"
SEED="${Q397_GRPO_SEED:-0}"
OUTPUT="results/blackbox/${METHOD}"
LOG_DIR="logs/lambda/${METHOD}"
mkdir -p "${LOG_DIR}" "${WANDB_DIR}"
LOG="${LOG_DIR}/run-$(date -u +%Y%m%dT%H%M%SZ).out"
exec > >(tee -a "${LOG}") 2>&1

echo "method=${METHOD}"
echo "generation_batch_size=${GENERATION_BATCH_SIZE} vllm_memory=${VLLM_MEMORY}"
echo "muon_lr=${MUON_LR} adamw_lr=${ADAMW_LR} epochs=${EPOCHS} temperature=${TEMPERATURE}"

accelerate launch \
  --num_processes 1 \
  --main_process_port 0 \
  experiments/qwen_grpo_lora/run_qwen_grpo_lora.py \
  --config-name qwen_grpo_lora_q397_optimized_reasoning \
  "method=${METHOD}" \
  "output_dir=${OUTPUT}" \
  "training.generation_batch_size=${GENERATION_BATCH_SIZE}" \
  "vllm.gpu_memory_utilization=${VLLM_MEMORY}" \
  "training.muon_learning_rate=${MUON_LR}" \
  "training.learning_rate=${ADAMW_LR}" \
  "training.num_train_epochs=${EPOCHS}" \
  "training.temperature=${TEMPERATURE}" \
  "training.max_completion_length=${MAX_COMPLETION_LENGTH}" \
  "evaluation.generate_completions=false" \
  "seed=${SEED}"

python experiments/qwen_grpo_lora/evaluate_qwen_grpo_lora_logits_vllm.py \
  --adapter-dir "${OUTPUT}/adapter" \
  --split validation \
  --output-dir "${OUTPUT}/validation_vllm_logits" \
  --prefix-variant prediction \
  --label-style plain \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 64
