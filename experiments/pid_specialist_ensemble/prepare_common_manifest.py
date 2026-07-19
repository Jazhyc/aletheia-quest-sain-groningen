#!/usr/bin/env python3
"""Prepare one leakage-auditable training subset shared by specialist caches."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.train_student_sft import (
    load_records,
    select_stratified_fraction,
)


def common_usable_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Return label-consistent records usable in every teacher cache."""
    by_source = []
    labels: dict[tuple[str, Any], int] = {}
    representative: dict[tuple[str, Any], dict[str, Any]] = {}
    for path in paths:
        records = load_records(path, dataset_name_contains="varied-deception")
        current = {}
        for record in records:
            key = (str(record["dataset"]), record["index"])
            label = int(record["label"])
            if key in labels and labels[key] != label:
                raise ValueError(f"teacher caches disagree on label for {key}")
            labels[key] = label
            representative[key] = record
            current[key] = record
        by_source.append(set(current))
    if not by_source:
        raise ValueError("at least one teacher cache is required")
    common = set.intersection(*by_source)
    return [representative[key] for key in sorted(common)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="append", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    common = common_usable_records([path.resolve() for path in args.cache])
    selected = select_stratified_fraction(common, args.fraction, args.seed)
    manifest = [
        {
            "dataset": str(record["dataset"]),
            "index": record["index"],
            "label": int(record["label"]),
        }
        for record in selected
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in manifest)
    )
    report = {
        "caches": [path.resolve().as_posix() for path in args.cache],
        "common_usable": len(common),
        "selected": len(selected),
        "fraction": args.fraction,
        "seed": args.seed,
        "selected_by_label": dict(sorted(Counter(
            int(record["label"]) for record in selected
        ).items())),
        "selected_by_stratum": {
            json.dumps([dataset, label]): count
            for (dataset, label), count in sorted(Counter(
                (str(record["dataset"]), int(record["label"]))
                for record in selected
            ).items())
        },
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
