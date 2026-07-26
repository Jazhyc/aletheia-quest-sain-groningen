#!/usr/bin/env python3
"""Summarize source grounding, abstention, and repeated-fact consistency."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def attach_labels(
    rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join labels only for diagnostics; labels never enter teacher prompts."""
    labels = {
        (str(row["question"]), str(row["answer_full"])): int(row["label"])
        for row in label_rows
    }
    return [
        {
            **row,
            "deceptive": labels.get((str(row["question"]), str(row["answer_full"]))),
        }
        for row in rows
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    polarities = Counter()
    comparisons = Counter()
    predicates: dict[str, Counter[str]] = defaultdict(Counter)
    by_hidden_class: dict[str, Counter[str]] = defaultdict(Counter)
    error_types = Counter()
    for row in rows:
        annotation = row.get("annotation") or {}
        polarity = annotation.get("polarity") or "abstain"
        comparison = annotation.get("comparison") or "missing"
        polarities[polarity] += 1
        comparisons[comparison] += 1
        predicates[row["candidate"]["predicate"]][polarity] += 1
        if row.get("deceptive") is not None:
            by_hidden_class[str(row["deceptive"])][polarity] += 1
        error_types.update(row.get("validation_errors", []))

    repeated: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        annotation = row.get("annotation") or {}
        key = (
            row["question_group"],
            row["candidate"].get("qid", ""),
            row["candidate"]["predicate"],
            row["candidate"]["fact"],
            str(annotation.get("claimed_value", "")).casefold().strip(),
        )
        repeated[key].append(annotation.get("polarity") or "abstain")
    repeated_groups = [values for values in repeated.values() if len(values) > 1]
    consistent_groups = sum(len(set(values)) == 1 for values in repeated_groups)

    return {
        "candidates": len(rows),
        "polarities": dict(polarities),
        "polar_rate": (
            (polarities["supports"] + polarities["contradicts"]) / max(1, len(rows))
        ),
        "comparisons": dict(comparisons),
        "validation_error_candidates": sum(bool(row.get("validation_errors")) for row in rows),
        "validation_error_types": dict(error_types),
        "polarity_vetoes": sum(
            bool((row.get("annotation") or {}).get("polarity_veto")) for row in rows
        ),
        "repeated_fact_groups": len(repeated_groups),
        "repeated_fact_exact_polarity_agreement": (
            consistent_groups / max(1, len(repeated_groups))
        ),
        "by_hidden_deception_class_diagnostic_only": {
            key: dict(value) for key, value in sorted(by_hidden_class.items())
        },
        "by_predicate": {
            key: dict(value)
            for key, value in sorted(
                predicates.items(),
                key=lambda item: (-sum(item[1].values()), item[0]),
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label-source", type=Path)
    args = parser.parse_args()
    label_rows = load(args.label_source) if args.label_source else []
    report = {
        path.stem: summarize(
            attach_labels(load(path), label_rows) if label_rows else load(path)
        )
        for path in args.inputs
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
