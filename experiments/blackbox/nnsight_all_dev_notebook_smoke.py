#!/usr/bin/env python
"""Run the submitted notebook locally over a small slice of every dry.yaml dataset."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml


def default_submission_notebook() -> str:
    notebooks = sorted(Path("submission").glob("*.ipynb"))
    if len(notebooks) != 1:
        raise RuntimeError(
            f"submission/ must contain exactly one notebook, found {len(notebooks)}"
        )
    return notebooks[0].as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dry.yaml")
    parser.add_argument("--notebook", default=default_submission_notebook())
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--summary", default="logs/nnsight_all_dev_smoke_summary.csv")
    return parser.parse_args()


def load_datasets(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    datasets = [str(entry["name"]) for entry in data.get("datasets", [])]
    if not datasets:
        raise RuntimeError(f"{path} has no datasets")
    return datasets


def main() -> None:
    args = parse_args()
    datasets = load_datasets(Path(args.config))
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    failures = 0
    for index, dataset in enumerate(datasets, start=1):
        print(f"[{index}/{len(datasets)}] {dataset}", flush=True)
        env = os.environ.copy()
        if args.batch_size is not None:
            env["PHOENIX_BATCH_SIZE"] = str(args.batch_size)
        cmd = [
            sys.executable,
            "experiments/blackbox/nnsight_local_notebook_smoke.py",
            "--notebook",
            args.notebook,
            "--dataset",
            dataset,
            "--limit",
            str(args.limit),
            "--timeout",
            str(args.timeout),
        ]
        proc = subprocess.run(cmd, env=env, text=True)
        ok = proc.returncode == 0
        if not ok:
            failures += 1
        rows.append({
            "dataset": dataset,
            "ok": str(ok),
            "returncode": str(proc.returncode),
        })
        print(f"[{index}/{len(datasets)}] {'ok' if ok else 'failed'}", flush=True)

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "ok", "returncode"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary_path}", flush=True)
    if failures:
        raise SystemExit(f"{failures} dataset smoke run(s) failed")


if __name__ == "__main__":
    main()
