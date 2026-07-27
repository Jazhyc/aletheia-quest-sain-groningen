#!/bin/bash
# Score continuous action-report margins in one local Slurm model load.

#SBATCH --job-name=aq-action-auroc
#SBATCH --time=00:45:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail

PHASE="${1:-}"
if [[ "${PHASE}" != "development" && "${PHASE}" != "confirmation" ]]; then
  echo "usage: $0 development|confirmation" >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
COMMON_GIT_DIR="$(git rev-parse --git-common-dir)"
SHARED_ROOT="$(cd "${COMMON_GIT_DIR}/.." && pwd)"

mkdir -p logs/slurm/liars_bench_distillation
LOG="logs/slurm/liars_bench_distillation/action-auroc-${PHASE}-${SLURM_JOB_ID}.out"
exec >"${LOG}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0
source "${SHARED_ROOT}/.venv/bin/activate"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

METHOD="liars_bench_action_report_auroc_v1"
ARTIFACT_ROOT="results/blackbox/${METHOD}"
EVAL_ARTIFACT="${SHARED_ROOT}/results/blackbox/liars_bench_ood_gptoss_v1/eval.jsonl"
ADAPTER="${SHARED_ROOT}/results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter"
EXTRA_ARGS=()
if [[ "${PHASE}" == "confirmation" ]]; then
  EXTRA_ARGS+=(--selection "${ARTIFACT_ROOT}/development/selection.json")
fi

python experiments/liars_bench_distillation/evaluate_action_report_auroc.py \
  --eval-artifact "${EVAL_ARTIFACT}" \
  --phoenix-adapter "${ADAPTER}" \
  --phoenix-config configs/liars_bench_prompt_control.yaml \
  --output-dir "${ARTIFACT_ROOT}/${PHASE}" \
  --split "${PHASE}" \
  --expected-rows 100 \
  --minimum-auroc-gain 0.05 \
  --maximum-source-auroc-loss 0.05 \
  "${EXTRA_ARGS[@]}"
