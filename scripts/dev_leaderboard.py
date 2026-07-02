#!/usr/bin/env python3
"""Run a submission on a local dev split and append private leaderboard records."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_SRC = ROOT / "leaderboard" / "src"
sys.path.insert(0, str(LEADERBOARD_SRC))

from aletheia_runner.config import DatasetConfig, METRIC_KEYS, RunnerConfig  # noqa: E402
from aletheia_runner.pipeline import run_pipeline  # noqa: E402
from aletheia_runner.results import summarize_submission  # noqa: E402


SPLIT_NAMES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".", help="Repository root containing submission/.")
    p.add_argument("--splits-dir", default="dev_splits",
                   help="Directory created by scripts/make_dev_splits.py.")
    p.add_argument("--split", choices=SPLIT_NAMES, default="validation",
                   help="Which local split to score.")
    p.add_argument("--method", required=True,
                   help="Method name to store in the private leaderboard.")
    p.add_argument("--tag", choices=("white", "black"), default=None,
                   help="Optional method category tag.")
    p.add_argument("--ledger", default=None,
                   help="JSONL ledger path; default: <splits-dir>/results.jsonl.")
    p.add_argument("--ndif-api-key", default=os.environ.get("NDIF_API_KEY"),
                   help="NDIF key forwarded to notebook runs.")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HF token forwarded to notebook runs.")
    p.add_argument("--limit", type=int, default=None,
                   help="Forward ALETHEIA_LIMIT for smoke runs. Not representative.")
    p.add_argument("--cache-dir", default=None,
                   help="Dataset cache dir; default uses a temp aletheia-dev-cache.")
    p.add_argument("--no-append", action="store_true",
                   help="Print results without appending to the ledger.")
    return p.parse_args()


def resolve_labels_uri(uri: str, root: Path, cfg_dir: Path) -> str:
    label_path = Path(uri)
    if label_path.is_absolute() or "://" in uri:
        return uri
    for base in (Path.cwd(), root, cfg_dir):
        candidate = base / label_path
        if candidate.exists():
            return candidate.resolve().as_posix()
    return (root / label_path).resolve().as_posix()


def load_config(path: Path, root: Path, cache_dir: str | None) -> RunnerConfig:
    data = yaml.safe_load(path.read_text()) or {}
    datasets = []
    for item in data.get("datasets", []):
        item = dict(item)
        item["labels_uri"] = resolve_labels_uri(item["labels_uri"], root, path.parent)
        datasets.append(DatasetConfig(**item))
    if not datasets:
        raise SystemExit(f"{path} has no datasets entries")
    return RunnerConfig(
        datasets=datasets,
        sandbox=True,
        confine=False,
        enforce_egress=False,
        redact_errors=False,
        score_partial=True,
        ndif_host=data.get("ndif_host", "https://aletheias.api.ndif.us"),
        notebook_timeout=int(data.get("notebook_timeout", 2700)),
        cache_dir=cache_dir or str(Path(tempfile.gettempdir()) / "aletheia-dev-cache"),
    )


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def fmt(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else "-"


def print_summary(record: dict) -> None:
    metrics = record["summary"]["metrics"]
    print(f"{record['method']} on {record['split']}: "
          f"balanced_accuracy={fmt(metrics.get('balanced_accuracy'))} "
          f"auroc={fmt(metrics.get('auroc'))}")
    print("Per-dataset:")
    for row in record["summary"]["datasets"]:
        print("  "
              f"{row['dataset']} | "
              f"bal={fmt(row.get('balanced_accuracy'))} "
              f"auc={fmt(row.get('auroc'))} "
              f"recall={fmt(row.get('recall'))} "
              f"fpr={fmt(row.get('fpr'))}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    splits_dir = Path(args.splits_dir)
    if not splits_dir.is_absolute():
        splits_dir = root / splits_dir
    cfg_path = splits_dir / f"dry.{args.split}.yaml"
    if not cfg_path.exists():
        raise SystemExit(
            f"{cfg_path} does not exist. Run scripts/make_dev_splits.py first."
        )

    extra_env = {}
    if args.ndif_api_key:
        extra_env["NDIF_API_KEY"] = args.ndif_api_key
    if args.hf_token:
        extra_env["HF_TOKEN"] = args.hf_token
    if args.limit is not None:
        extra_env["ALETHEIA_LIMIT"] = str(args.limit)

    config = load_config(cfg_path, root, args.cache_dir)
    records = run_pipeline(
        root,
        args.method,
        config,
        extra_env=extra_env or None,
        on_progress=lambda nb, ds, phase: print(f"[{phase}] {nb} :: {ds}"),
    )
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for rec in records:
        rec.submitted_at = now
        rec.tag = args.tag

    summary = summarize_submission(records)
    record = {
        "submitted_at": now,
        "method": args.method,
        "split": args.split,
        "tag": args.tag,
        "metrics": {k: summary["metrics"].get(k) for k in METRIC_KEYS},
        "summary": summary,
        "records": [asdict(r) for r in records],
    }
    print_summary(record)

    if not args.no_append:
        ledger = Path(args.ledger) if args.ledger else splits_dir / "results.jsonl"
        append_jsonl(ledger, record)
        print(f"Appended {ledger}")


if __name__ == "__main__":
    main()
