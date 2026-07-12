#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")/../.."

if command -v module >/dev/null 2>&1; then
  module load Python/3.12.3-GCCcore-13.3.0
fi
source .venv/bin/activate
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

mkdir -p logs/ndif/fact_extractor_self_consistency
LOG_PATH="logs/ndif/fact_extractor_self_consistency/whole-output-validation-$(date +%Y%m%d-%H%M%S).log"
exec >"${LOG_PATH}" 2>&1

export NDIF_HOST="${NDIF_HOST:-https://aletheias.api.ndif.us}"
export TOKENIZERS_PARALLELISM=false
python experiments/fact_extractor_self_consistency/run_ndif_whole_output_self_query.py
