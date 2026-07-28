#!/bin/bash
# Benchmark effective-batch-32 Qwen-397B soft-student training on Lambda H100.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ -f "${HOME}/.config/aletheia/runtime.env" ]]; then
  source "${HOME}/.config/aletheia/runtime.env"
fi
source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

BENCHMARK="qwen397_soft_distillation_speed_benchmark_v1"
LOG_DIR="logs/lambda/${BENCHMARK}"
CACHE_ROOT="results/blackbox/qwen35_397b_fp8_nothink_truth_value_binary_logit_v1/train"
BASE_CONFIG="pid_qwen397_tvg_binary_soft_distillation_v1"
MAX_STEPS="${Q397_BENCHMARK_STEPS:-8}"
mkdir -p "${LOG_DIR}"

sha256sum -c "${CACHE_ROOT}/SHA256SUMS"

run_condition() {
  local name="$1"
  local batch_size="$2"
  local accumulation="$3"
  local compile="$4"
  local method="${BENCHMARK}_${name}"
  local output="results/blackbox/${method}"
  local log="${LOG_DIR}/${name}.out"

  echo "benchmark name=${name} batch=${batch_size} accumulation=${accumulation} compile=${compile}"
  rm -f "${log}"
  python experiments/privileged_information_distillation/train_student_sft.py \
    --config-name "${BASE_CONFIG}" \
    "method=${method}" \
    "output_dir=${output}" \
    "student.output_dir=${output}/adapter" \
    "student.training.max_steps=${MAX_STEPS}" \
    "student.training.per_device_train_batch_size=${batch_size}" \
    "student.training.gradient_accumulation_steps=${accumulation}" \
    "student.training.torch_compile=${compile}" \
    2>&1 | tee "${log}"
}

batch_sizes=(2 4 8 16)
accumulations=(16 8 4 2)
conditions=()
for index in "${!batch_sizes[@]}"; do
  name="b${batch_sizes[$index]}_e32_eager"
  if run_condition \
      "${name}" \
      "${batch_sizes[$index]}" \
      "${accumulations[$index]}" \
      false; then
    conditions+=("${name}")
  else
    echo "benchmark condition failed; continuing: ${name}" >&2
  fi
done
if [[ "${#conditions[@]}" -eq 0 ]]; then
  echo "all eager benchmark conditions failed" >&2
  exit 1
fi

read -r best_batch best_accumulation < <(
  python experiments/privileged_information_distillation/summarize_qwen397_speed_benchmark.py \
    --log-dir "${LOG_DIR}" \
    --best-shell \
    "${conditions[@]/#/--condition=}"
)

compile_name="b${best_batch}_e32_compile"
if run_condition "${compile_name}" "${best_batch}" "${best_accumulation}" true; then
  conditions+=("${compile_name}")
else
  echo "compile benchmark failed; retaining eager results" >&2
fi

python experiments/privileged_information_distillation/summarize_qwen397_speed_benchmark.py \
  --log-dir "${LOG_DIR}" \
  --output "${LOG_DIR}/summary.json" \
  "${conditions[@]/#/--condition=}"
