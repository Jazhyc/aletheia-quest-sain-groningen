#!/usr/bin/env bash
# Create a local development environment for Aletheia's Quest.
#
# Builds a Python 3.12 UV environment at ./.venv with everything needed to
# develop your method against the public dev data, train probes with GPU tooling,
# and run `python submit.py --dry` (the real leaderboard pipeline, locally).
#
#   ./setup_dev.sh            # create/sync ./.venv
#   source .venv/bin/activate # then work in it, or use `uv run ...`
#   python submit.py --dry
#
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: uv is required for the development environment.

Install it with:
  curl -LsSf https://astral.sh/uv/install.sh | sh

or see https://docs.astral.sh/uv/getting-started/installation/
EOF
  exit 1
fi

VENV="${VENV:-.venv}"
export UV_PROJECT_ENVIRONMENT="$VENV"
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

echo "Syncing UV environment at $VENV ..."
uv sync

cat <<'EOF'

Dev environment ready.

  source .venv/bin/activate              # or prefix commands with `UV_CACHE_DIR=.uv-cache uv run`
  export NDIF_API_KEY="your-ndif-key"     # from your competition signup
  huggingface-cli login                   # so HF model configs/tokenizers load
  python submit.py --dry                  # rehearse on the datasets in dry.yaml

EOF
