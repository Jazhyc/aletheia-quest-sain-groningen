#!/usr/bin/env python3
"""Apply the frozen family-coverage adapter acceptance rule."""

from __future__ import annotations

import argparse
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
from experiments.model_specific_readout_routing.analyze_adapter_route import (
    family_from_dataset,
)


def load_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["dataset"]), str(row["index"])): row
        for row in map(json.loads, path.read_text().splitlines())
    }


def group_ba(
    rows: list[dict[str, Any]], predictions: list[int]
) -> dict[str, float]:
    result = {"all": macro_dataset_ba(rows, predictions)}
    for family in ("Qwen", "Gemma", "Nemotron"):
        offsets = [
            offset
            for offset, row in enumerate(rows)
            if family_from_dataset(str(row["dataset"])) == family
        ]
        result[family] = macro_dataset_ba(
            [rows[offset] for offset in offsets],
            [predictions[offset] for offset in offsets],
        )
    qwen_varied = [
        offset
        for offset, row in enumerate(rows)
        if family_from_dataset(str(row["dataset"])) == "Qwen"
        and "varied-deception" in str(row["dataset"])
    ]
    result["Qwen_varied"] = macro_dataset_ba(
        [rows[offset] for offset in qwen_varied],
        [predictions[offset] for offset in qwen_varied],
    )
    return result


def analyze(
    baseline_rows: dict[tuple[str, str], dict[str, Any]],
    candidate_rows: dict[tuple[str, str], dict[str, Any]],
    *,
    max_overall_loss: float = 0.0025,
    max_qwen_varied_loss: float = 0.005,
) -> dict[str, Any]:
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("baseline and candidate rows differ")
    keys = sorted(baseline_rows)
    rows = [baseline_rows[key] for key in keys]
    baseline = [parsed_prediction(baseline_rows[key]) for key in keys]
    candidate = [parsed_prediction(candidate_rows[key]) for key in keys]
    baseline_ba = group_ba(rows, baseline)
    candidate_ba = group_ba(rows, candidate)
    deltas = {name: candidate_ba[name] - value for name, value in baseline_ba.items()}
    fixes = breaks = 0
    for row, old, new in zip(rows, baseline, candidate, strict=True):
        if old == new:
            continue
        label = int(row["label"])
        fixes += int(new == label)
        breaks += int(old == label)
    accepted = (
        deltas["all"] >= -max_overall_loss
        and deltas["Gemma"] >= 0.0
        and deltas["Nemotron"] >= 0.0
        and deltas["Qwen_varied"] >= -max_qwen_varied_loss
    )
    return {
        "criteria": {
            "max_overall_loss": max_overall_loss,
            "max_qwen_varied_loss": max_qwen_varied_loss,
            "minimum_gemma_delta": 0.0,
            "minimum_nemotron_delta": 0.0,
        },
        "baseline_ba": baseline_ba,
        "candidate_ba": candidate_ba,
        "deltas": deltas,
        "changes": sum(a != b for a, b in zip(baseline, candidate, strict=True)),
        "fixes": fixes,
        "breaks": breaks,
        "accepted": accepted,
        "test_evaluation_authorized": accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(load_rows(args.baseline), load_rows(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
