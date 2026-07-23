#!/usr/bin/env python3
"""Compare fixed binary vote rules from cached ensemble generations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    """Return balanced accuracy, recall, and false-positive rate."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("each metric stratum must contain both labels")
    true_positives = sum(y == 1 and p == 1 for y, p in zip(labels, predictions, strict=True))
    false_positives = sum(y == 0 and p == 1 for y, p in zip(labels, predictions, strict=True))
    recall = true_positives / positives
    fpr = false_positives / negatives
    return {
        "balanced_accuracy": (recall + 1.0 - fpr) / 2.0,
        "recall": recall,
        "fpr": fpr,
    }


def aggregate_generations(
    rows: list[dict[str, Any]],
    *,
    stratum_field: str,
) -> dict[str, Any]:
    """Evaluate each member plus OR, majority, and unanimous vote rules."""
    labels: dict[tuple[str, str], int] = {}
    strata: dict[tuple[str, str], str] = {}
    votes: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    parse_errors: dict[str, int] = defaultdict(int)
    for row in rows:
        key = (str(row["dataset"]), str(row["index"]))
        member = str(row["ensemble_member"])
        labels[key] = int(row["label"])
        strata[key] = str(row[stratum_field])
        rating = row.get("rating")
        votes[key][member] = int(rating is not None and int(rating) >= 2)
        parse_errors[member] += int(bool(row.get("parse_error")))

    members = sorted({member for member_votes in votes.values() for member in member_votes})
    if len(members) < 2:
        raise ValueError("expected at least two ensemble members")
    if any(set(member_votes) != set(members) for member_votes in votes.values()):
        raise ValueError("incomplete ensemble rows")

    grouped_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, stratum in strata.items():
        grouped_keys[stratum].append(key)

    rule_names = [*members, "or", "majority", "unanimous"]
    by_rule: dict[str, Any] = {}
    for rule_name in rule_names:
        by_stratum = {}
        for stratum, keys in sorted(grouped_keys.items()):
            predictions = []
            for key in keys:
                member_votes = votes[key]
                if rule_name in member_votes:
                    prediction = member_votes[rule_name]
                elif rule_name == "or":
                    prediction = int(any(member_votes.values()))
                elif rule_name == "majority":
                    prediction = int(sum(member_votes.values()) > len(members) / 2)
                else:
                    prediction = int(all(member_votes.values()))
                predictions.append(prediction)
            by_stratum[stratum] = binary_metrics([labels[key] for key in keys], predictions)
        by_rule[rule_name] = {
            "macro": {
                metric: sum(values[metric] for values in by_stratum.values()) / len(by_stratum)
                for metric in ("balanced_accuracy", "recall", "fpr")
            },
            "by_stratum": by_stratum,
        }

    return {
        "n": len(labels),
        "stratum_field": stratum_field,
        "n_strata": len(grouped_keys),
        "members": members,
        "parse_errors_by_member": dict(sorted(parse_errors.items())),
        "disagreement_rows": sum(len(set(member_votes.values())) > 1 for member_votes in votes.values()),
        "rules": by_rule,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--stratum-field", default="dataset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.generations.read_text().splitlines()]
    result = aggregate_generations(rows, stratum_field=args.stratum_field)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
