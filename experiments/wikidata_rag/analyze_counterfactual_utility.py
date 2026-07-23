#!/usr/bin/env python3
"""Summarize downstream utility, shuffled controls, and semantic relevance labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "positive_gt_0": int((array > 0.0).sum()),
        "positive_gt_0_01": int((array > 0.01).sum()),
        "harmful_lt_minus_0_01": int((array < -0.01).sum()),
    }


def binary_change(empty_score: float, candidate_score: float, label: int) -> int:
    before = int((empty_score >= 0.5) == bool(label))
    after = int((candidate_score >= 0.5) == bool(label))
    return after - before


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = []
    controlled = []
    shuffled = []
    semantic: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"utility": [], "controlled_utility": []}
    )
    binary_changes = []
    best_raw = []
    best_controlled = []
    best_raw_binary = []
    best_controlled_binary = []
    rows_with_candidates = 0
    rows_with_semantic = 0
    for row in rows:
        candidates = row["candidates"]
        shuffled.append(float(row["shuffled_utility"]))
        rows_with_candidates += int(bool(candidates))
        rows_with_semantic += int(any("semantic_label" in item for item in candidates))
        for candidate in candidates:
            raw.append(float(candidate["utility"]))
            controlled.append(float(candidate["controlled_utility"]))
            binary_changes.append(binary_change(
                float(row["empty_score"]), float(candidate["score"]), int(row["label"])
            ))
            if "semantic_label" in candidate:
                group = semantic[candidate["semantic_label"]]
                group["utility"].append(float(candidate["utility"]))
                group["controlled_utility"].append(
                    float(candidate["controlled_utility"])
                )
        if candidates:
            raw_choice = max(candidates, key=lambda item: item["utility"])
            controlled_choice = max(candidates, key=lambda item: item["controlled_utility"])
            best_raw.append(float(raw_choice["utility"]))
            best_controlled.append(float(controlled_choice["controlled_utility"]))
            best_raw_binary.append(binary_change(
                float(row["empty_score"]), float(raw_choice["score"]), int(row["label"])
            ))
            best_controlled_binary.append(binary_change(
                float(row["empty_score"]), float(controlled_choice["score"]), int(row["label"])
            ))
    return {
        "rows": len(rows),
        "rows_with_candidates": rows_with_candidates,
        "rows_with_semantic_labels": rows_with_semantic,
        "candidate_utility": distribution(raw),
        "candidate_controlled_utility": distribution(controlled),
        "shuffled_utility": distribution(shuffled),
        "candidate_binary_changes": {
            "rescues": binary_changes.count(1),
            "harms": binary_changes.count(-1),
            "unchanged": binary_changes.count(0),
        },
        "row_oracle_raw_utility": distribution(best_raw),
        "row_oracle_controlled_utility": distribution(best_controlled),
        "row_oracle_raw_binary_changes": {
            "rescues": best_raw_binary.count(1), "harms": best_raw_binary.count(-1)
        },
        "row_oracle_controlled_binary_changes": {
            "rescues": best_controlled_binary.count(1),
            "harms": best_controlled_binary.count(-1),
        },
        "by_semantic_label": {
            label: {
                name: distribution(values) for name, values in measurements.items()
            }
            for label, measurements in sorted(semantic.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        path.stem: summarize(load_jsonl(path)) for path in args.input
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
