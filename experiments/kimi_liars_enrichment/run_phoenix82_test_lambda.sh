#!/bin/bash
# Run the frozen Phoenix 8.1 versus Phoenix 8.2 candidate test comparison.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

full="results/blackbox/qwen9b_kimi_k3_tvg_soft_full_plus_liars_r16_lr5e5_ep2_v1/adapter"
candidate="results/blackbox/kimi_k3_liars_enrichment_interpolation_v1/full75/adapter"

python experiments/privileged_information_distillation/evaluate_student_sft.py \
  --split test \
  --run-name test_phoenix82_frozen \
  --max-new-tokens 1 \
  --continuous-margins \
  --continuous-margin-condition direct \
  --adapter-dir "${full}" \
  --adapter-dir "${candidate}"
