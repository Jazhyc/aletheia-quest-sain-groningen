#!/usr/bin/env python3
"""Paired uncertainty audit for a frozen heavy-judge capacity swap."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from experiments.liars_bench_distillation.analyze_semantic_heavy_hybrid import (
    paired_bootstrap_deltas,
)
from experiments.liars_bench_distillation.evaluate_students import balanced_accuracy


def analyze(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    samples: int = 20_000,
    seed: int = 20260717,
) -> dict[str, Any]:
    baseline_keys = [
        (str(row["dataset"]), str(row["index"]), int(row["label"]))
        for row in baseline
    ]
    candidate_keys = [
        (str(row["dataset"]), str(row["index"]), int(row["label"]))
        for row in candidate
    ]
    if baseline_keys != candidate_keys:
        raise ValueError("baseline/candidate row identities differ")
    changes: dict[str, dict[str, int]] = defaultdict(
        lambda: {"changed": 0, "fixes": 0, "breaks": 0}
    )
    for base, cand in zip(baseline, candidate, strict=True):
        if int(base["prediction"]) == int(cand["prediction"]):
            continue
        category = str(base["category"])
        changes[category]["changed"] += 1
        if int(cand["prediction"]) == int(base["label"]):
            changes[category]["fixes"] += 1
        else:
            changes[category]["breaks"] += 1
    totals = {
        key: sum(value[key] for value in changes.values())
        for key in ("changed", "fixes", "breaks")
    }
    return {
        "n": len(baseline),
        **totals,
        "changes_by_category": dict(sorted(changes.items())),
        "paired_label_stratified_bootstrap": paired_bootstrap_deltas(
            baseline,
            candidate,
            samples=samples,
            seed=seed,
        ),
    }


def parse_partition(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    baseline_generations: list[dict[str, Any]],
    candidate_generations: list[dict[str, Any]],
) -> dict[str, Any]:
    def error_counts(rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for row in rows:
            key = (str(row["dataset"]), str(row["index"]))
            counts[key] += int(bool(row.get("parse_error")))
        return counts

    baseline_errors = error_counts(baseline_generations)
    candidate_errors = error_counts(candidate_generations)
    groups: dict[str, list[dict[str, Any]]] = {
        "baseline_clean": [],
        "baseline_any_parse_error": [],
    }
    changes = {
        "baseline_clean": {"fixes": 0, "breaks": 0},
        "baseline_any_parse_error": {"fixes": 0, "breaks": 0},
    }
    for base, cand in zip(baseline, candidate, strict=True):
        key = (str(base["dataset"]), str(base["index"]))
        group = (
            "baseline_any_parse_error"
            if baseline_errors.get(key, 0)
            else "baseline_clean"
        )
        groups[group].append({
            "label": int(base["label"]),
            "baseline_prediction": int(base["prediction"]),
            "candidate_prediction": int(cand["prediction"]),
        })
        if int(base["prediction"]) != int(cand["prediction"]):
            outcome = (
                "fixes"
                if int(cand["prediction"]) == int(base["label"])
                else "breaks"
            )
            changes[group][outcome] += 1

    partitions = {}
    for group, rows in groups.items():
        base_rows = [
            {"label": row["label"], "prediction": row["baseline_prediction"]}
            for row in rows
        ]
        candidate_rows = [
            {"label": row["label"], "prediction": row["candidate_prediction"]}
            for row in rows
        ]
        partitions[group] = {
            "n": len(rows),
            "baseline": balanced_accuracy(base_rows),
            "candidate": balanced_accuracy(candidate_rows),
            **changes[group],
        }
    return {
        "baseline_rows_with_any_parse_error": sum(
            count > 0 for count in baseline_errors.values()
        ),
        "baseline_rows_with_all_members_parse_error": sum(
            count == 3 for count in baseline_errors.values()
        ),
        "candidate_rows_with_any_parse_error": sum(
            count > 0 for count in candidate_errors.values()
        ),
        "candidate_rows_with_all_members_parse_error": sum(
            count == 3 for count in candidate_errors.values()
        ),
        "partitions": partitions,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--baseline-generations", type=Path)
    parser.add_argument("--candidate-generations", type=Path)
    args = parser.parse_args()
    result = analyze(
        baseline := load_jsonl(args.baseline),
        candidate := load_jsonl(args.candidate),
        samples=args.samples,
        seed=args.seed,
    )
    if (args.baseline_generations is None) != (args.candidate_generations is None):
        raise ValueError("provide both generation caches or neither")
    if args.baseline_generations is not None:
        result["parse_error_partition"] = parse_partition(
            baseline,
            candidate,
            load_jsonl(args.baseline_generations),
            load_jsonl(args.candidate_generations),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
