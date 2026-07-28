#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python experiments/apollo_system_framing_counterfactual/run.py "$@"
