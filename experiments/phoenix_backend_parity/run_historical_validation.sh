#!/bin/bash
#SBATCH --job-name=phoenix-v21-validation-replay
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail

ROOT="/scratch/s4626451/Aletheias-Quest-Competition"
cd "${ROOT}"

LOG_DIR="logs/slurm/phoenix_backend_parity"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/validation-replay-${SLURM_JOB_ID}.out"
exec >"${LOG_FILE}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source .venv/bin/activate

SOURCE_ROOT="results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1"
CONVERTED_ROOT="results/blackbox/qwen35_peft_path_migration_20260728/repositories/Jazhyc--aletheias-phoenix-wright-v2-1-adapter/converted"
REPLAY_ROOT="results/blackbox/phoenix_v21_current_loader_validation_replay_v1"
REPLAY_ADAPTER="${REPLAY_ROOT}/adapter"

mkdir -p "${REPLAY_ADAPTER}"
cp -f "${SOURCE_ROOT}/config.yaml" "${REPLAY_ROOT}/config.yaml"
cp -f "${SOURCE_ROOT}/adapter/chat_template.jinja" "${REPLAY_ADAPTER}/chat_template.jinja"
cp -f "${SOURCE_ROOT}/adapter/tokenizer.json" "${REPLAY_ADAPTER}/tokenizer.json"
cp -f "${SOURCE_ROOT}/adapter/tokenizer_config.json" "${REPLAY_ADAPTER}/tokenizer_config.json"
cp -f "${CONVERTED_ROOT}/adapter_config.json" "${REPLAY_ADAPTER}/adapter_config.json"
cp -f "${CONVERTED_ROOT}/adapter_model.safetensors" "${REPLAY_ADAPTER}/adapter_model.safetensors"

python experiments/privileged_information_distillation/evaluate_student_sft.py \
    --adapter-dir "${REPLAY_ADAPTER}" \
    --split validation \
    --run-name current_loader_validation \
    --continuous-margins \
    --continuous-margin-condition direct \
    --max-new-tokens 1 \
    --gpu-memory-utilization 0.82
