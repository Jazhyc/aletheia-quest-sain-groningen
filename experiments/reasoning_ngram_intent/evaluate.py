#!/usr/bin/env python3
"""Evaluate a frozen reasoning-only n-gram artifact on one local split."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reasoning_ngram_intent.train import (  # noqa: E402
    Candidate,
    load_reasoning_split,
    metric_summary,
    score_model,
    write_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=ROOT / "dev_splits",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = joblib.load(args.artifact)
    candidate = Candidate(**artifact["candidate"])
    frame = load_reasoning_split(args.split, args.splits_dir.resolve())
    scores = score_model(
        frame,
        candidate,
        artifact["vectorizer"],
        artifact["classifier"],
    )
    report = {
        "split": args.split,
        "artifact": str(args.artifact.resolve()),
        "candidate": dataclasses.asdict(candidate),
        "metrics": metric_summary(frame, scores),
        "rows": len(frame),
        "datasets": int(frame["dataset"].nunique()),
        "empty_reasoning": int(frame["reasoning"].eq("").sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_predictions(args.output.with_suffix(".csv"), frame, scores)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
