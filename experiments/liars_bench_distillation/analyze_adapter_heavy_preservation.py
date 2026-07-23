#!/usr/bin/env python3
"""Gate the Phoenix-LoRA heavy default against matched base-Qwen outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare_adapter_heavy(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    if baseline["n"] != 800 or candidate["n"] != 800:
        raise ValueError("expected matched 800-row spectrum results")
    category_deltas = {
        key: (
            candidate["by_category"][key]["balanced_accuracy"]
            - baseline["by_category"][key]["balanced_accuracy"]
        )
        for key in baseline["by_category"]
    }
    cell_deltas = {
        key: (
            candidate["by_category_family"][key]["balanced_accuracy"]
            - baseline["by_category_family"][key]["balanced_accuracy"]
        )
        for key in baseline["by_category_family"]
    }
    limits = {
        "minimum_macro_delta": -0.01,
        "minimum_category_delta": -0.02,
        "minimum_category_family_delta": -0.05,
        "maximum_parse_errors": 181,
    }
    metrics = {
        "baseline_macro_category_ba": baseline["macro_category_ba"],
        "candidate_macro_category_ba": candidate["macro_category_ba"],
        "macro_delta": (
            candidate["macro_category_ba"] - baseline["macro_category_ba"]
        ),
        "category_deltas": category_deltas,
        "category_family_deltas": cell_deltas,
        "candidate_parse_errors": int(candidate["parse_errors"]),
    }
    passed = bool(
        metrics["macro_delta"] >= limits["minimum_macro_delta"]
        and min(category_deltas.values()) >= limits["minimum_category_delta"]
        and min(cell_deltas.values()) >= limits["minimum_category_family_delta"]
        and metrics["candidate_parse_errors"] <= limits["maximum_parse_errors"]
    )
    return {**metrics, "gate": {**limits, "passed": passed}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_adapter_heavy(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
