#!/usr/bin/env python3
"""Evaluate a training-coverage-selected Phoenix adapter route by family."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.model_specific_readout_routing.analyze import (
    macro_dataset_ba,
    parsed_prediction,
)


def family_from_dataset(dataset: str) -> str:
    """Map a public competition unit to its source-model family."""
    lowered = dataset.casefold()
    if "qwen" in lowered:
        return "Qwen"
    if "gemma" in lowered:
        return "Gemma"
    if "nemotron" in lowered:
        return "Nemotron"
    raise ValueError(f"unknown source-model family in {dataset!r}")


def load_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["dataset"]), str(row["index"])): row
        for row in map(json.loads, path.read_text().splitlines())
    }


def route_predictions(
    varied_rows: dict[tuple[str, str], dict[str, Any]],
    mixed_rows: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int], list[int], list[int]]:
    """Use varied-only weights for Qwen and mixed weights for other families."""
    if varied_rows.keys() != mixed_rows.keys():
        missing = len(varied_rows.keys() - mixed_rows.keys())
        extra = len(mixed_rows.keys() - varied_rows.keys())
        raise ValueError(f"adapter rows differ: missing={missing} extra={extra}")
    keys = sorted(varied_rows)
    rows = [varied_rows[key] for key in keys]
    varied = [parsed_prediction(varied_rows[key]) for key in keys]
    mixed = [parsed_prediction(mixed_rows[key]) for key in keys]
    routed = [
        varied_prediction
        if family_from_dataset(str(row["dataset"])) == "Qwen"
        else mixed_prediction
        for row, varied_prediction, mixed_prediction in zip(
            rows, varied, mixed, strict=True
        )
    ]
    return rows, varied, mixed, routed


def summarize_split(
    varied_path: Path,
    mixed_path: Path,
) -> dict[str, Any]:
    rows, varied, mixed, routed = route_predictions(
        load_rows(varied_path), load_rows(mixed_path)
    )
    family_offsets: dict[str, list[int]] = defaultdict(list)
    for offset, row in enumerate(rows):
        family_offsets[family_from_dataset(str(row["dataset"]))].append(offset)
    fixes = breaks = 0
    for row, baseline, candidate in zip(rows, varied, routed, strict=True):
        if baseline == candidate:
            continue
        label = int(row["label"])
        fixes += int(candidate == label)
        breaks += int(baseline == label)
    return {
        "varied_only_ba": macro_dataset_ba(rows, varied),
        "mixed_ba": macro_dataset_ba(rows, mixed),
        "routed_ba": macro_dataset_ba(rows, routed),
        "routed_delta_vs_varied_only": (
            macro_dataset_ba(rows, routed) - macro_dataset_ba(rows, varied)
        ),
        "changes": sum(a != b for a, b in zip(varied, routed, strict=True)),
        "fixes": fixes,
        "breaks": breaks,
        "per_family_routed_ba": {
            family: macro_dataset_ba(
                [rows[offset] for offset in offsets],
                [routed[offset] for offset in offsets],
            )
            for family, offsets in sorted(family_offsets.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for split in ("validation", "test"):
        parser.add_argument(f"--{split}-varied", type=Path, required=True)
        parser.add_argument(f"--{split}-mixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "route": {
            "Qwen": "varied_only",
            "Gemma": "mixed",
            "Nemotron": "mixed",
        },
        "selection_basis": "training-data family coverage; no validation labels",
        "splits": {
            split: summarize_split(
                getattr(args, f"{split}_varied"),
                getattr(args, f"{split}_mixed"),
            )
            for split in ("validation", "test")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
