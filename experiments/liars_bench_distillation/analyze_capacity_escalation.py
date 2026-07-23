#!/usr/bin/env python3
"""Evaluate a fixed Q9-disagreement trigger for selective Q27 escalation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from experiments.liars_bench_distillation.analyze_ensemble_votes import binary_metrics
from experiments.liars_bench_distillation.evaluate_heavy_spectrum import source_family


Key = tuple[str, str]


def load_votes(rows: list[dict[str, Any]]) -> tuple[
    dict[Key, dict[str, int]], dict[Key, int], dict[Key, dict[str, str]]
]:
    """Load member binary votes with the production negative parse fallback."""
    votes: dict[Key, dict[str, int]] = defaultdict(dict)
    labels: dict[Key, int] = {}
    metadata: dict[Key, dict[str, str]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["index"]))
        rating = row.get("rating")
        votes[key][str(row["ensemble_member"])] = int(
            rating is not None and int(rating) >= 2
        )
        labels[key] = int(row["label"])
        metadata[key] = {
            "dataset": str(row["dataset"]),
            "category": str(row.get("category", row["dataset"])),
            "source_family": source_family(str(row["source_model"]))
            if row.get("source_model")
            else "unknown",
        }
    members = {frozenset(member_votes) for member_votes in votes.values()}
    if len(members) != 1 or len(next(iter(members))) < 2:
        raise ValueError("expected the same multi-member ensemble for every row")
    return votes, labels, metadata


def macro_metrics(
    labels: dict[Key, int],
    predictions: dict[Key, int],
    metadata: dict[Key, dict[str, str]],
    *,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    """Compute metric strata and their macro average."""
    groups: dict[str, list[Key]] = defaultdict(list)
    for key in labels:
        group = "/".join(metadata[key][field] for field in fields)
        groups[group].append(key)
    by_group = {
        group: binary_metrics(
            [labels[key] for key in keys],
            [predictions[key] for key in keys],
        )
        for group, keys in sorted(groups.items())
    }
    return {
        "macro": {
            metric: sum(values[metric] for values in by_group.values()) / len(by_group)
            for metric in ("balanced_accuracy", "recall", "fpr")
        },
        "by_group": by_group,
    }


def analyze(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    macro_field: str,
    minimum_macro_delta: float,
    minimum_group_delta: float,
    minimum_category_family_delta: float,
    maximum_query_fraction: float,
) -> dict[str, Any]:
    """Apply the frozen disagreement cascade and its preservation gate."""
    baseline_votes, labels, metadata = load_votes(baseline_rows)
    candidate_votes, candidate_labels, candidate_metadata = load_votes(candidate_rows)
    if labels != candidate_labels or set(baseline_votes) != set(candidate_votes):
        raise ValueError("baseline and candidate rows or labels differ")
    if metadata != candidate_metadata:
        raise ValueError("baseline and candidate metadata differ")

    baseline_predictions = {
        key: int(any(member_votes.values())) for key, member_votes in baseline_votes.items()
    }
    candidate_predictions = {
        key: int(any(member_votes.values())) for key, member_votes in candidate_votes.items()
    }
    triggers = {
        key: len(set(member_votes.values())) > 1
        for key, member_votes in baseline_votes.items()
    }
    cascade_predictions = {
        key: candidate_predictions[key] if triggers[key] else baseline_predictions[key]
        for key in baseline_predictions
    }

    condition_metrics = {
        name: macro_metrics(labels, predictions, metadata, fields=(macro_field,))
        for name, predictions in {
            "baseline": baseline_predictions,
            "candidate": candidate_predictions,
            "cascade": cascade_predictions,
        }.items()
    }
    family_metrics = {
        name: macro_metrics(
            labels,
            predictions,
            metadata,
            fields=("category", "source_family"),
        )
        for name, predictions in {
            "baseline": baseline_predictions,
            "cascade": cascade_predictions,
        }.items()
    }
    baseline_macro = condition_metrics["baseline"]["macro"]["balanced_accuracy"]
    cascade_macro = condition_metrics["cascade"]["macro"]["balanced_accuracy"]
    group_deltas = {
        group: values["balanced_accuracy"]
        - condition_metrics["baseline"]["by_group"][group]["balanced_accuracy"]
        for group, values in condition_metrics["cascade"]["by_group"].items()
    }
    category_family_deltas = {
        group: values["balanced_accuracy"]
        - family_metrics["baseline"]["by_group"][group]["balanced_accuracy"]
        for group, values in family_metrics["cascade"]["by_group"].items()
    }
    query_fraction = sum(triggers.values()) / len(triggers)
    macro_delta = cascade_macro - baseline_macro
    passed = bool(
        macro_delta >= minimum_macro_delta
        and min(group_deltas.values()) >= minimum_group_delta
        and min(category_family_deltas.values()) >= minimum_category_family_delta
        and query_fraction <= maximum_query_fraction
    )
    return {
        "n": len(labels),
        "trigger": "baseline_member_binary_disagreement",
        "q27_query_rows": sum(triggers.values()),
        "q27_query_fraction": query_fraction,
        "changed_rows": sum(
            cascade_predictions[key] != baseline_predictions[key]
            for key in baseline_predictions
        ),
        "conditions": condition_metrics,
        "macro_ba_delta": macro_delta,
        "group_ba_deltas": group_deltas,
        "category_family_ba_deltas": category_family_deltas,
        "gate": {
            "minimum_macro_delta": minimum_macro_delta,
            "minimum_group_delta": minimum_group_delta,
            "minimum_category_family_delta": minimum_category_family_delta,
            "maximum_query_fraction": maximum_query_fraction,
            "passed": passed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-generations", type=Path, required=True)
    parser.add_argument("--candidate-generations", type=Path, required=True)
    parser.add_argument("--macro-field", default="category")
    parser.add_argument("--minimum-macro-delta", type=float, default=0.02)
    parser.add_argument("--minimum-group-delta", type=float, default=-0.02)
    parser.add_argument("--minimum-category-family-delta", type=float, default=-0.05)
    parser.add_argument("--maximum-query-fraction", type=float, default=0.35)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        [json.loads(line) for line in args.baseline_generations.read_text().splitlines()],
        [json.loads(line) for line in args.candidate_generations.read_text().splitlines()],
        macro_field=args.macro_field,
        minimum_macro_delta=args.minimum_macro_delta,
        minimum_group_delta=args.minimum_group_delta,
        minimum_category_family_delta=args.minimum_category_family_delta,
        maximum_query_fraction=args.maximum_query_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
