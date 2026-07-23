#!/usr/bin/env python3
"""Analyze fixed-threshold and per-dataset rank calibration of Phoenix margins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import macro_metrics


SCORE_COLUMNS = ("empty_margin_score", "reasoning_margin_score")


def scenario_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    result = {"all": macro_metrics(frame, 0.5)}
    for scenario in ("instructed", "varied"):
        subset = frame[frame["dataset"].str.contains(scenario)]
        result[scenario] = macro_metrics(subset, 0.5)
    return result


def rank_fraction_predictions(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
) -> pd.DataFrame:
    """Replace scores by deterministic top-fraction predictions per dataset."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    pieces = []
    for _, group in frame.groupby("dataset", sort=True):
        group = group.copy()
        count = round(fraction * len(group))
        order = np.lexsort((group["index"].to_numpy(), -group[score_column].to_numpy()))
        predictions = np.zeros(len(group), dtype=float)
        predictions[order[:count]] = 1.0
        group["score"] = predictions
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def best_threshold(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    candidates = np.linspace(0.0, 1.0, 101)
    scored = frame[["dataset", "label", score_column]].rename(columns={score_column: "score"})
    results = [macro_metrics(scored, float(threshold))["balanced_accuracy"] for threshold in candidates]
    best_index = int(np.argmax(results))
    return {
        "threshold": float(candidates[best_index]),
        "balanced_accuracy": float(results[best_index]),
    }


def analyze(frame: pd.DataFrame) -> dict[str, object]:
    output: dict[str, object] = {}
    generated = frame["prediction"].fillna(0).astype(bool)
    for score_column in SCORE_COLUMNS:
        scores = frame[score_column]
        ordinary = frame[["dataset", "label", score_column]].rename(
            columns={score_column: "score"}
        )
        output[score_column] = {
            "unique_scores": int(scores.nunique()),
            "quantiles": {
                str(quantile): float(value)
                for quantile, value in scores.quantile(
                    [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
                ).items()
            },
            "fixed_half": scenario_metrics(ordinary),
            "top_half": scenario_metrics(
                rank_fraction_predictions(frame, score_column, 0.5)
            ),
            "best_split_threshold": best_threshold(frame, score_column),
            "agreement_with_generated": float(((scores >= 0.5) == generated).mean()),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generations", type=Path)
    args = parser.parse_args()
    frame = pd.read_json(args.generations, lines=True)
    print(json.dumps(analyze(frame), indent=2))


if __name__ == "__main__":
    main()
