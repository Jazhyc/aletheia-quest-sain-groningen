#!/usr/bin/env python3
"""Select a prompt using frozen Liars' Bench development/confirmation gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.analyze_transfer import (
    external_delta,
    paired_changes_grouped,
    passes_external_floors,
)


def select_prompt(
    result: dict[str, Any],
    *,
    baseline_name: str,
    minimum_macro_gain: float,
    minimum_category_delta: float,
    minimum_category_source_model_delta: float,
    maximum_parse_error_increase: int,
    generation_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare prompt conditions and select the best condition that passes."""
    conditions = result["conditions"]
    if baseline_name not in conditions:
        raise ValueError(f"missing baseline condition {baseline_name!r}")
    baseline = conditions[baseline_name]
    comparisons: dict[str, Any] = {}
    passing = []
    for name, candidate in conditions.items():
        if name == baseline_name:
            continue
        delta = external_delta(candidate, baseline)
        parse_error_increase = (
            int(candidate["parse_errors"]) - int(baseline["parse_errors"])
        )
        passes = bool(
            delta["macro_delta"] >= minimum_macro_gain
            and parse_error_increase <= maximum_parse_error_increase
            and passes_external_floors(
                delta,
                minimum_category_delta=minimum_category_delta,
                minimum_source_model_delta=None,
                minimum_category_source_model_delta=(
                    minimum_category_source_model_delta
                ),
            )
        )
        comparison: dict[str, Any] = {
            "external": delta,
            "parse_error_increase": parse_error_increase,
            "passes": passes,
        }
        if generation_dir is not None:
            comparison["paired"] = paired_changes_grouped(
                generation_dir / f"{baseline_name}.jsonl",
                generation_dir / f"{name}.jsonl",
            )
        comparisons[name] = comparison
        if passes:
            passing.append(name)
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda name: (
                comparisons[name]["external"]["macro_delta"],
                min(
                    comparisons[name]["external"][
                        "category_source_model_deltas"
                    ].values()
                ),
                -comparisons[name]["parse_error_increase"],
                name,
            ),
        )
    return {
        "split": result.get("split"),
        "baseline": baseline_name,
        "criteria": {
            "minimum_macro_gain": minimum_macro_gain,
            "minimum_category_delta": minimum_category_delta,
            "minimum_category_source_model_delta": (
                minimum_category_source_model_delta
            ),
            "maximum_parse_error_increase": maximum_parse_error_increase,
        },
        "selected": selected,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path)
    parser.add_argument("--baseline", default="control")
    parser.add_argument("--minimum-macro-gain", type=float, required=True)
    parser.add_argument("--minimum-category-delta", type=float, default=-0.02)
    parser.add_argument(
        "--minimum-category-source-model-delta",
        type=float,
        default=-0.05,
    )
    parser.add_argument("--maximum-parse-error-increase", type=int, default=10)
    args = parser.parse_args()
    report = select_prompt(
        json.loads(args.result.read_text()),
        baseline_name=args.baseline,
        minimum_macro_gain=args.minimum_macro_gain,
        minimum_category_delta=args.minimum_category_delta,
        minimum_category_source_model_delta=(
            args.minimum_category_source_model_delta
        ),
        maximum_parse_error_increase=args.maximum_parse_error_increase,
        generation_dir=args.generation_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
