#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python experiments/ndif_tvg_model_swap/run_ndif_matched_organism_tvg.py "$@"
