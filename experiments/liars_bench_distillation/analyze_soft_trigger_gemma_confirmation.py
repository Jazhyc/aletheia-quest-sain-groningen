#!/usr/bin/env python3
"""Apply the frozen family-route confirmation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def analyze(
    external: dict[str, Any],
    baseline_competition: dict[str, Any],
    candidate_competition: dict[str, Any],
) -> dict[str, Any]:
    baseline = external["conditions"]["baseline"]
    candidate = external["conditions"]["candidate"]
    external_delta = (
        candidate["metrics"]["balanced_accuracy"]
        - baseline["metrics"]["balanced_accuracy"]
    )
    source_deltas = {
        source: (
            candidate["per_source_model"][source]["balanced_accuracy"]
            - baseline["per_source_model"][source]["balanced_accuracy"]
        )
        for source in baseline["per_source_model"]
    }
    gemma_datasets = [
        name for name in baseline_competition["per_dataset"]
        if "gemma" in name.lower()
    ]
    competition_deltas = {
        name: (
            candidate_competition["per_dataset"][name]["balanced_accuracy"]
            - baseline_competition["per_dataset"][name]["balanced_accuracy"]
        )
        for name in gemma_datasets
    }
    parse_delta = candidate["parse_errors"] - baseline["parse_errors"]
    passed = bool(
        external_delta >= 0.01
        and min(source_deltas.values()) >= -0.02
        and min(competition_deltas.values()) >= -0.005
        and parse_delta <= 2
    )
    return {
        "external_baseline": baseline,
        "external_candidate": candidate,
        "external_ba_delta": external_delta,
        "source_model_ba_deltas": source_deltas,
        "competition_gemma_ba_deltas": competition_deltas,
        "parse_error_delta": parse_delta,
        "gate": {
            "minimum_external_ba_gain": 0.01,
            "minimum_source_model_ba_delta": -0.02,
            "minimum_competition_gemma_ba_delta": -0.005,
            "maximum_parse_error_increase": 2,
            "passed": passed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--baseline-competition", type=Path, required=True)
    parser.add_argument("--candidate-competition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        json.loads(args.external.read_text()),
        json.loads(args.baseline_competition.read_text()),
        json.loads(args.candidate_competition.read_text()),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
