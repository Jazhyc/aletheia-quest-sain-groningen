#!/usr/bin/env python3
"""Combine a frozen heavy default with explicit semantic-report specialists."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.liars_bench_distillation.evaluate_students import balanced_accuracy


SPECIALIST_BY_CATEGORY = {
    "harm-pressure-knowledge-report": "epistemic",
    "insider-trading": "action",
}


def category_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    return {
        category: balanced_accuracy(group)
        for category, group in sorted(grouped.items())
    }


def category_family_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['category']}/{row['source_family']}"
        grouped[key].append(row)
    return {key: balanced_accuracy(group) for key, group in sorted(grouped.items())}


def paired_bootstrap_deltas(
    heavy_rows: list[dict[str, Any]],
    routed_rows: list[dict[str, Any]],
    *,
    samples: int = 20_000,
    seed: int = 20260717,
) -> dict[str, Any]:
    """Class-stratified paired bootstrap for category and macro BA deltas."""
    if len(heavy_rows) != len(routed_rows):
        raise ValueError("paired bootstrap rows differ in length")
    rng = np.random.default_rng(seed)
    category_samples = {}
    categories = sorted({str(row["category"]) for row in heavy_rows})
    for category in categories:
        indices = [
            index
            for index, row in enumerate(heavy_rows)
            if str(row["category"]) == category
        ]
        labels = np.asarray([int(heavy_rows[index]["label"]) for index in indices])
        deltas = np.asarray([
            int(routed_rows[index]["prediction"])
            - int(heavy_rows[index]["prediction"])
            for index in indices
        ])
        positive = deltas[labels == 1]
        negative = deltas[labels == 0]
        if not len(positive) or not len(negative):
            raise ValueError(f"category {category!r} lacks both labels")
        sampled_positive = positive[
            rng.integers(0, len(positive), size=(samples, len(positive)))
        ].mean(axis=1)
        sampled_negative = negative[
            rng.integers(0, len(negative), size=(samples, len(negative)))
        ].mean(axis=1)
        category_samples[category] = (sampled_positive - sampled_negative) / 2.0
    macro = np.vstack(list(category_samples.values())).mean(axis=0)

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
            "probability_positive": float((values > 0).mean()),
        }

    return {
        "samples": samples,
        "macro": summary(macro),
        "by_category": {
            category: summary(values)
            for category, values in category_samples.items()
        },
    }


def analyze(
    heavy_rows: list[dict[str, Any]],
    epistemic_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(heavy_rows) != 800:
        raise ValueError("expected the frozen 800-row heavy spectrum")
    specialist_maps = {
        "epistemic": {str(row["index"]): row for row in epistemic_rows},
        "action": {str(row["index"]): row for row in action_rows},
    }
    routed = []
    fixes = breaks = changed = 0
    route_counts: dict[str, int] = defaultdict(int)
    for row in heavy_rows:
        specialist_name = SPECIALIST_BY_CATEGORY.get(str(row["category"]))
        prediction = int(row["prediction"])
        if specialist_name is not None:
            specialist = specialist_maps[specialist_name].get(str(row["index"]))
            if specialist is None:
                raise ValueError(
                    f"missing {specialist_name} prediction for {row['index']}"
                )
            if int(specialist["label"]) != int(row["label"]):
                raise ValueError("specialist/heavy labels differ")
            prediction = int(specialist["prediction"])
            route_counts[specialist_name] += 1
        if prediction != int(row["prediction"]):
            changed += 1
            if prediction == int(row["label"]):
                fixes += 1
            else:
                breaks += 1
        routed.append({**row, "prediction": prediction})

    heavy_category = category_metrics(heavy_rows)
    routed_category = category_metrics(routed)
    heavy_cells = category_family_metrics(heavy_rows)
    hybrid_cells = category_family_metrics(routed)
    cell_deltas = {
        key: (
            hybrid_cells[key]["balanced_accuracy"]
            - heavy_cells[key]["balanced_accuracy"]
        )
        for key in heavy_cells
    }
    category_deltas = {
        category: (
            routed_category[category]["balanced_accuracy"]
            - heavy_category[category]["balanced_accuracy"]
        )
        for category in heavy_category
    }
    heavy_macro = sum(
        value["balanced_accuracy"] for value in heavy_category.values()
    ) / len(heavy_category)
    routed_macro = sum(
        value["balanced_accuracy"] for value in routed_category.values()
    ) / len(routed_category)
    gate = {
        "minimum_macro_ba_gain": 0.03,
        "minimum_category_ba_delta": 0.0,
        "macro_ba_gain": routed_macro - heavy_macro,
        "category_ba_deltas": category_deltas,
    }
    gate["passed"] = bool(
        gate["macro_ba_gain"] >= gate["minimum_macro_ba_gain"]
        and min(category_deltas.values()) >= gate["minimum_category_ba_delta"]
    )
    return {
        "n": len(routed),
        "route_counts": dict(route_counts),
        "heavy_macro_category_ba": heavy_macro,
        "hybrid_macro_category_ba": routed_macro,
        "heavy_by_category": heavy_category,
        "hybrid_by_category": routed_category,
        "category_family_ba_deltas": cell_deltas,
        "worst_category_family_ba_delta": min(cell_deltas.values()),
        "paired_bootstrap": paired_bootstrap_deltas(heavy_rows, routed),
        "changed": changed,
        "fixes": fixes,
        "breaks": breaks,
        "gate": gate,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heavy", type=Path, required=True)
    parser.add_argument("--epistemic", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        load_jsonl(args.heavy),
        load_jsonl(args.epistemic),
        load_jsonl(args.action),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
