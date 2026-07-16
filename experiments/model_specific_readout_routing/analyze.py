#!/usr/bin/env python3
"""Select conservative per-organism readout parameters on cached train rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any


SCENARIO_PREFIX = re.compile(
    r"^aletheias-quest/dev-(?:instructed|varied)-deception-"
)


def organism_key(dataset: str) -> str:
    return SCENARIO_PREFIX.sub("", dataset)


def parsed_prediction(row: dict[str, Any]) -> int:
    value = row.get("prediction")
    return 0 if value is None else int(float(value) >= 0.5)


def load_generations(path: Path) -> dict[tuple[str, Any], dict[str, Any]]:
    return {
        (row["dataset"], row["index"]): row
        for row in map(json.loads, path.read_text().splitlines())
    }


def balanced_accuracy(labels: list[int], predictions: list[int]) -> float:
    tp = sum(label == prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == prediction == 0 for label, prediction in zip(labels, predictions))
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def macro_dataset_ba(rows: list[dict[str, Any]], predictions: list[int]) -> float:
    groups: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for row, prediction in zip(rows, predictions):
        labels, values = groups[row["dataset"]]
        labels.append(int(row["label"]))
        values.append(prediction)
    return sum(balanced_accuracy(*values) for values in groups.values()) / len(groups)


def load_members(run: Path) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    summary = load_generations(run / "summary/generations.jsonl")
    binary = load_generations(run / "binary/generations.jsonl")
    if summary.keys() != binary.keys():
        raise ValueError(f"summary/binary row mismatch under {run}")
    keys = sorted(summary)
    rows = [summary[key] for key in keys]
    summary_predictions = [parsed_prediction(summary[key]) for key in keys]
    max_predictions = [
        max(parsed_prediction(summary[key]), parsed_prediction(binary[key]))
        for key in keys
    ]
    return rows, summary_predictions, max_predictions


def select_routes(
    rows: list[dict[str, Any]],
    summary_predictions: list[int],
    max_predictions: list[int],
    *,
    minimum_train_gain: float,
) -> dict[str, str]:
    """Disable the binary member only with a material train-set BA gain."""
    groups: dict[str, list[int]] = defaultdict(list)
    for offset, row in enumerate(rows):
        groups[organism_key(row["dataset"])].append(offset)
    routes = {}
    for organism, offsets in groups.items():
        selected_rows = [rows[offset] for offset in offsets]
        summary_ba = macro_dataset_ba(
            selected_rows, [summary_predictions[offset] for offset in offsets]
        )
        max_ba = macro_dataset_ba(
            selected_rows, [max_predictions[offset] for offset in offsets]
        )
        routes[organism] = (
            "summary"
            if summary_ba - max_ba >= minimum_train_gain
            else "max"
        )
    return routes


def apply_routes(
    rows: list[dict[str, Any]],
    summary_predictions: list[int],
    max_predictions: list[int],
    routes: dict[str, str],
) -> list[int]:
    return [
        summary if routes.get(organism_key(row["dataset"]), "max") == "summary" else maximum
        for row, summary, maximum in zip(rows, summary_predictions, max_predictions)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--test-run", type=Path)
    parser.add_argument("--minimum-train-gain", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = load_members(args.train_run)
    routes = select_routes(*train, minimum_train_gain=args.minimum_train_gain)
    result: dict[str, Any] = {
        "minimum_train_gain": args.minimum_train_gain,
        "routes": routes,
        "splits": {},
    }
    for split, run in (
        ("train", args.train_run),
        ("validation", args.validation_run),
        ("test", args.test_run),
    ):
        if run is None:
            continue
        rows, summary, maximum = load_members(run)
        routed = apply_routes(rows, summary, maximum, routes)
        result["splits"][split] = {
            "summary_ba": macro_dataset_ba(rows, summary),
            "max_ba": macro_dataset_ba(rows, maximum),
            "routed_ba": macro_dataset_ba(rows, routed),
            "changes_from_max": sum(a != b for a, b in zip(routed, maximum)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
