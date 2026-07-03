#!/bin/bash
#SBATCH --job-name=aq-blackbox
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm

if command -v module >/dev/null 2>&1; then
  module load Python/3.12.3-GCCcore-13.3.0
  module load CUDA/13.2.0
fi

source .venv/bin/activate

resolve_run_label() {
  python - "$@" <<'PY'
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

args = sys.argv[1:]
config_name = "blackbox_judge"
config_path = "configs"
overrides = []
flags_with_values = {
    "--cfg",
    "--config-dir",
    "--config-name",
    "--config-path",
    "--experimental-rerun",
    "--package",
}

i = 0
while i < len(args):
    arg = args[i]
    if arg == "--config-name" and i + 1 < len(args):
        config_name = args[i + 1]
        i += 2
        continue
    if arg.startswith("--config-name="):
        config_name = arg.split("=", 1)[1]
        i += 1
        continue
    if arg == "--config-path" and i + 1 < len(args):
        config_path = args[i + 1]
        i += 2
        continue
    if arg.startswith("--config-path="):
        config_path = arg.split("=", 1)[1]
        i += 1
        continue
    if arg in flags_with_values:
        i += 2
        continue
    if not arg.startswith("-"):
        overrides.append(arg)
    i += 1

config_dir = Path(config_path)
if not config_dir.is_absolute():
    config_dir = Path.cwd() / config_dir

with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg = compose(config_name=config_name, overrides=overrides)

print(f"{cfg.get('method', 'unknown')}\t{cfg.get('split', 'unknown')}")
PY
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if RUN_LABEL="$(resolve_run_label "$@")"; then
    METHOD="$(printf '%s' "${RUN_LABEL}" | cut -f1)"
    SPLIT="$(printf '%s' "${RUN_LABEL}" | cut -f2)"
  else
    METHOD="unknown"
    SPLIT="unknown"
  fi

  SAFE_METHOD="$(printf '%s' "${METHOD}" | tr -cs 'A-Za-z0-9_.=-' '_')"
  SAFE_SPLIT="$(printf '%s' "${SPLIT}" | tr -cs 'A-Za-z0-9_.=-' '_')"
  METHOD_LOG_DIR="logs/slurm/${SAFE_METHOD}"
  METHOD_LOG_FILE="${METHOD_LOG_DIR}/${SAFE_SPLIT}-${SLURM_JOB_ID}.out"
  BOOTSTRAP_LOG_FILE="logs/slurm/${SLURM_JOB_NAME:-aq-blackbox}-${SLURM_JOB_ID}.bootstrap.out"
  mkdir -p "${METHOD_LOG_DIR}"
  echo "Redirecting Slurm job output to ${METHOD_LOG_FILE}"
  exec >"${METHOD_LOG_FILE}" 2>&1
  rm -f "${BOOTSTRAP_LOG_FILE}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "method=${METHOD}"
  echo "split=${SPLIT}"
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

python experiments/blackbox/run_judge.py "$@"
