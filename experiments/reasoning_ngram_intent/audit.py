#!/usr/bin/env python3
"""Audit a frozen reasoning n-gram model without fitting or selecting anything."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pid_specialist_ensemble.analyze_ensemble import (  # noqa: E402
    binary_metrics,
)
from experiments.reasoning_ngram_intent.train import (  # noqa: E402
    Candidate,
    INTENT_CUE_RE,
    load_reasoning_split,
    metric_summary,
    score_model,
)


def per_dataset(frame, scores: np.ndarray) -> list[dict]:
    scored = frame[["dataset", "label"]].copy()
    scored["score"] = scores
    return [
        {
            "dataset": dataset,
            "rows": len(group),
            **binary_metrics(
                group["label"].to_numpy(dtype=int),
                group["score"].to_numpy(dtype=float),
            ),
        }
        for dataset, group in scored.groupby("dataset", sort=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["validation", "test"],
        default=["validation", "test"],
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=ROOT / "dev_splits",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = joblib.load(args.artifact)
    candidate = Candidate(**artifact["candidate"])
    report = {
        "artifact": str(args.artifact.resolve()),
        "candidate": dataclasses.asdict(candidate),
        "threshold": float(artifact["threshold"]),
        "selection": "none; read-only post-selection audit",
        "splits": {},
    }
    for split in args.splits:
        frame = load_reasoning_split(split, args.splits_dir.resolve())
        scores = score_model(
            frame,
            candidate,
            artifact["vectorizer"],
            artifact["classifier"],
        )
        redacted_scores = score_model(
            frame,
            candidate,
            artifact["vectorizer"],
            artifact["classifier"],
            redact_intent_cues=True,
        )
        explicit_cue_scores = np.asarray([
            float(bool(INTENT_CUE_RE.search(reasoning)))
            for reasoning in frame["reasoning"]
        ])
        report["splits"][split] = {
            "model": metric_summary(frame, scores),
            "model_per_dataset": per_dataset(frame, scores),
            "explicit_cue_only": metric_summary(frame, explicit_cue_scores),
            "intent_cues_redacted": metric_summary(frame, redacted_scores),
            "redaction_score_shift": {
                "mean_absolute": float(np.mean(np.abs(scores - redacted_scores))),
                "max_absolute": float(np.max(np.abs(scores - redacted_scores))),
                "binary_changes": int(
                    ((scores >= 0.5) != (redacted_scores >= 0.5)).sum()
                ),
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
