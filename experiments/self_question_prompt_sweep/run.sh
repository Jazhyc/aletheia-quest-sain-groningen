#!/bin/bash
# Run the full-reasoning prompt sweep through competition NDIF.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if command -v module >/dev/null 2>&1; then
  module load Python/3.12.3-GCCcore-13.3.0
  module load CUDA/13.2.0
fi
source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

python experiments/self_question_prompt_sweep/run.py "$@"
