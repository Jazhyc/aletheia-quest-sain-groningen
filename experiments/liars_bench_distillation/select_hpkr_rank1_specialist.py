#!/usr/bin/env python3
"""Apply frozen development or confirmation gates to rank-1 HP-KR specialists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def cell_score(metrics: dict[str, Any]) -> tuple[str, float]:
    """Use BA for two-class cells and accuracy for structural one-class cells."""
    if metrics.get("balanced_accuracy") is not None:
        return "balanced_accuracy", float(metrics["balanced_accuracy"])
    return "accuracy", float(metrics["accuracy"])


def paired(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
) -> dict[str, int]:
    """Count changed decisions, fixes, and breaks by stable row key."""
    by_key = {
        (str(row["dataset"]), str(row["index"])): row
        for row in baseline
    }
    counts = {"changes": 0, "fixes": 0, "breaks": 0}
    for row in candidate:
        other = by_key[(str(row["dataset"]), str(row["index"]))]
        if int(row["prediction"]) == int(other["prediction"]):
            continue
        counts["changes"] += 1
        candidate_correct = int(row["prediction"]) == int(row["label"])
        baseline_correct = int(other["prediction"]) == int(other["label"])
        counts["fixes"] += int(candidate_correct and not baseline_correct)
        counts["breaks"] += int(baseline_correct and not candidate_correct)
    return counts


def compare(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    minimum_gain: float,
    minimum_source_delta: float,
    maximum_parse_error_increase: int,
) -> dict[str, Any]:
    """Compare a candidate to the frozen base-Qwen epistemic baseline."""
    source_deltas = {}
    source_metrics = {}
    for source, baseline_metrics in baseline["per_source_model"].items():
        metric, baseline_score = cell_score(baseline_metrics)
        candidate_metric, candidate_score = cell_score(
            candidate["per_source_model"][source]
        )
        if metric != candidate_metric:
            raise ValueError(f"metric mismatch for source {source}")
        source_metrics[source] = metric
        source_deltas[source] = candidate_score - baseline_score
    gain = (
        float(candidate["balanced_accuracy"])
        - float(baseline["balanced_accuracy"])
    )
    parse_increase = int(candidate["parse_errors"]) - int(baseline["parse_errors"])
    return {
        "balanced_accuracy": float(candidate["balanced_accuracy"]),
        "gain": gain,
        "source_deltas": source_deltas,
        "source_metrics": source_metrics,
        "parse_error_increase": parse_increase,
        "passes": bool(
            gain >= minimum_gain
            and min(source_deltas.values()) >= minimum_source_delta
            and parse_increase <= maximum_parse_error_increase
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--minimum-gain", type=float, required=True)
    parser.add_argument("--minimum-source-delta", type=float, default=-0.05)
    parser.add_argument("--maximum-parse-error-increase", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    conditions = result["conditions"]
    baseline_name = "base_epistemic"
    baseline = conditions[baseline_name]
    baseline_rows = [
        json.loads(line)
        for line in (
            args.generation_dir / f"{baseline_name}.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    comparisons = {}
    for name in args.candidate:
        comparison = compare(
            conditions[name],
            baseline,
            minimum_gain=args.minimum_gain,
            minimum_source_delta=args.minimum_source_delta,
            maximum_parse_error_increase=args.maximum_parse_error_increase,
        )
        candidate_rows = [
            json.loads(line)
            for line in (
                args.generation_dir / f"{name}.jsonl"
            ).read_text().splitlines()
            if line.strip()
        ]
        comparison["paired"] = paired(candidate_rows, baseline_rows)
        comparisons[name] = comparison

    passing = [name for name in args.candidate if comparisons[name]["passes"]]
    selected = max(
        passing,
        key=lambda name: (
            comparisons[name]["balanced_accuracy"],
            -int(conditions[name]["parse_errors"]),
            -args.candidate.index(name),
        ),
        default=None,
    )
    report = {
        "split": result["split"],
        "baseline": baseline_name,
        "reference": {
            "name": "phoenix_truthful_alternative",
            "balanced_accuracy": conditions["phoenix_truthful_alternative"][
                "balanced_accuracy"
            ],
        },
        "criteria": {
            "minimum_gain": args.minimum_gain,
            "minimum_source_delta": args.minimum_source_delta,
            "maximum_parse_error_increase": args.maximum_parse_error_increase,
        },
        "selected": selected,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
