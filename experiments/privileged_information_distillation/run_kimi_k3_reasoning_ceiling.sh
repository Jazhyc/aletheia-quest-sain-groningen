#!/bin/bash
# Benchmark Kimi K3 on trace-bearing varied test rows for a diagnostic ceiling.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

METHOD="kimi_k3_fireworks_reasoning4000_tvg_binary_logit_v1"
PYTHON="${ROOT}/.venv/bin/python"
CONCURRENCY="${KIMI_CONCURRENCY:-16}"

"${PYTHON}" -m experiments.openrouter_qwen397_tvg.run_openrouter_tvg \
  --model moonshotai/kimi-k3 \
  --method "${METHOD}" \
  --split test \
  --dataset-name-contains varied-deception \
  --include-reasoning \
  --reasoning-max-chars 4000 \
  --reasoning-truncation tail \
  --concurrency "${CONCURRENCY}" \
  --max-retries 8 \
  --provider-only Fireworks \
  --no-allow-fallbacks
