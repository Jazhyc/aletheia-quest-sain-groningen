#!/usr/bin/env python3
"""Extract comparable Trainer throughput from Qwen-397B student benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CONDITION_RE = re.compile(
    r"^b(?P<batch>\d+)_e32_(?P<backend>eager|compile)$"
)
METRIC_RE = re.compile(
    r"'train_runtime':\s*'?(?P<runtime>[0-9.]+)'?.*?"
    r"'train_samples_per_second':\s*'?(?P<samples>[0-9.]+)'?.*?"
    r"'train_steps_per_second':\s*'?(?P<steps>[0-9.]+)'?",
    re.S,
)


def load_row(log_dir: Path, condition: str) -> dict[str, object]:
    match = CONDITION_RE.fullmatch(condition)
    if match is None:
        raise ValueError(f"unexpected benchmark condition: {condition}")
    text = (log_dir / f"{condition}.out").read_text(errors="replace")
    metric_matches = list(METRIC_RE.finditer(text))
    if not metric_matches:
        raise ValueError(f"training metrics missing from {condition}.out")
    metrics = metric_matches[-1]
    batch = int(match.group("batch"))
    return {
        "condition": condition,
        "batch_size": batch,
        "gradient_accumulation_steps": 32 // batch,
        "backend": match.group("backend"),
        "train_runtime": float(metrics.group("runtime")),
        "train_samples_per_second": float(metrics.group("samples")),
        "train_steps_per_second": float(metrics.group("steps")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--best-shell", action="store_true")
    args = parser.parse_args()

    rows = sorted(
        (load_row(args.log_dir, condition) for condition in args.condition),
        key=lambda row: (-row["train_samples_per_second"], row["condition"]),
    )
    best = rows[0]
    if args.best_shell:
        print(best["batch_size"], best["gradient_accumulation_steps"])
        return
    print(json.dumps(rows, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
