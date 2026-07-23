#!/usr/bin/env python3
"""Diagnose pair-judge rating calibration without querying another model."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import macro_metrics


DEFAULT_INPUT = (
    ROOT
    / "results/blackbox/neutral_contrast_gpt_oss_pair_dks4096_generated_validation_v2/generations.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "calibration_diagnostic.json"
MEMBERS = ("details4096", "known4096", "scrutiny4096")
RATING_THRESHOLDS = tuple(range(2, 8))


def read_rating_frame(path: Path) -> pd.DataFrame:
    """Max-aggregate member ratings into one row per validation example."""
    with path.open() as handle:
        rows = pd.DataFrame(json.loads(line) for line in handle if line.strip())
    if rows["parse_error"].any():
        raise ValueError("calibration diagnostic requires fully parsed ratings")
    frame = rows.pivot(
        index=["neutral_variant", "dataset", "index", "label"],
        columns="ensemble_member",
        values="rating",
    ).reset_index()
    frame["score"] = (frame[list(MEMBERS)].max(axis=1) - 1.0) / 6.0
    return frame


def metrics_at_rating(frame: pd.DataFrame, rating: int) -> dict[str, Any]:
    """Evaluate a generated-rating cutoff using dataset-macro metrics."""
    return macro_metrics(frame[["dataset", "label", "score"]], (rating - 1.0) / 6.0)


def grouped_calibration(frame: pd.DataFrame) -> dict[str, Any]:
    """Select a cutoff without each held-out dataset unit's labels."""
    frame = frame.reset_index(drop=True)
    oof_predictions = np.zeros(len(frame), dtype=float)
    selections: list[dict[str, Any]] = []
    for dataset, held_out in frame.groupby("dataset", sort=True):
        train = frame[frame["dataset"] != dataset]
        candidates = [
            (metrics_at_rating(train, rating)["balanced_accuracy"], rating)
            for rating in RATING_THRESHOLDS
        ]
        _, selected_rating = max(candidates)  # Higher cutoff wins an exact BA tie.
        mask = frame["dataset"] == dataset
        threshold = (selected_rating - 1.0) / 6.0
        oof_predictions[mask] = (frame.loc[mask, "score"] >= threshold).astype(float)
        selections.append({
            "held_out_dataset": dataset,
            "held_out_rows": len(held_out),
            "selected_minimum_rating": selected_rating,
        })

    scored = frame[["dataset", "label"]].copy()
    scored["score"] = oof_predictions
    scenarios: dict[str, Any] = {}
    scenario_names = frame["dataset"].map(
        lambda name: "varied-deception" if "varied-deception" in name else "instructed-deception"
    )
    for name, group in scored.groupby(scenario_names, sort=True):
        scenarios[name] = macro_metrics(group, 0.5)
    histogram = Counter(item["selected_minimum_rating"] for item in selections)
    return {
        "protocol": "leave-one-dataset-unit-out",
        "tie_break": "higher rating cutoff",
        "threshold_histogram": {str(key): value for key, value in sorted(histogram.items())},
        "overall": macro_metrics(scored, 0.5),
        "scenarios": scenarios,
        "folds": selections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = read_rating_frame(args.input)
    variants: dict[str, Any] = {}
    for variant, group in frame.groupby("neutral_variant", sort=True):
        variants[variant] = {
            "threshold_sweep": {
                str(rating): metrics_at_rating(group, rating)
                for rating in RATING_THRESHOLDS
            },
            "grouped_calibration": grouped_calibration(group),
        }
    result = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(args.input.relative_to(ROOT)),
        "status": "diagnostic_only_no_test_authorization",
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
