#!/usr/bin/env python3
"""Select a Kimi pairwise scale from forward/reverse internal-holdout scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def direct_macro_auroc(result_path: Path) -> float:
    result = json.loads(result_path.read_text())
    return float(result["score_metrics"]["direct_margin"]["all"]["auroc"])


def select_scale(
    candidates: list[tuple[float, Path]],
    run_names: list[str],
    *,
    tie_tolerance: float,
) -> dict[str, Any]:
    """Average adapter-order controls and prefer less ranking weight near a tie."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    if len(run_names) < 2:
        raise ValueError("at least two adapter-order runs are required")
    if tie_tolerance < 0:
        raise ValueError("tie tolerance must be non-negative")

    rows = []
    for weight, adapter_dir in candidates:
        scores = [
            direct_macro_auroc(adapter_dir.parent / run_name / "result.json")
            for run_name in run_names
        ]
        rows.append({
            "weight": weight,
            "adapter_dir": adapter_dir.resolve().as_posix(),
            "scores": scores,
            "mean_auroc": sum(scores) / len(scores),
            "order_range": max(scores) - min(scores),
        })
    best_mean = max(row["mean_auroc"] for row in rows)
    eligible = [
        row
        for row in rows
        if best_mean - row["mean_auroc"] <= tie_tolerance
    ]
    selected = min(eligible, key=lambda row: row["weight"])
    return {
        "metric": "mean internal-holdout macro AUROC",
        "run_names": run_names,
        "tie_tolerance": tie_tolerance,
        "candidates": rows,
        "best_mean_auroc": best_mean,
        "selected_weight": selected["weight"],
        "selected_adapter_dir": selected["adapter_dir"],
        "selected_mean_auroc": selected["mean_auroc"],
        "selected_by_conservative_tie_break": (
            selected["mean_auroc"] < best_mean
        ),
    }


def parse_candidate(value: str) -> tuple[float, Path]:
    weight, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("candidate must be WEIGHT=ADAPTER_DIR")
    try:
        parsed_weight = float(weight)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid candidate weight: {weight}") from error
    if parsed_weight < 0:
        raise argparse.ArgumentTypeError("candidate weight must be non-negative")
    return parsed_weight, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
    )
    parser.add_argument("--run-name", action="append", required=True)
    parser.add_argument("--tie-tolerance", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = select_scale(
        args.candidate,
        args.run_name,
        tie_tolerance=args.tie_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
