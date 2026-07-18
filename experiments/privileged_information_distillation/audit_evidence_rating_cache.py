#!/usr/bin/env python3
"""Audit privileged evidence-rating teacher targets before student training."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


LEAK_PATTERNS = {
    "private_marker": re.compile(r"</?think>", re.IGNORECASE),
    "ground_truth": re.compile(r"\bground[- ]truth\b", re.IGNORECASE),
    "privileged_information": re.compile(
        r"\bprivileged (?:information|label|knowledge)\b", re.IGNORECASE
    ),
    "teacher_meta": re.compile(
        r"\b(?:as (?:a|the) teacher|being a teacher)\b", re.IGNORECASE
    ),
    "rubric_meta": re.compile(r"\b(?:the|this) rubric\b", re.IGNORECASE),
    "instruction_meta": re.compile(r"\bthese instructions\b", re.IGNORECASE),
}


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise RuntimeError(f"non-object record at {path}:{line_number}")
        records.append(record)
    return records


def rating_entropy(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count
    )


def audit_records(
    records: list[dict[str, Any]],
    *,
    expected_total: int,
    minimum_usable: int,
    minimum_distinct_ratings: int,
    minimum_intermediate_fraction: float,
) -> dict[str, Any]:
    errors: list[str] = []
    rating_counts: Counter[int] = Counter()
    rating_counts_by_label: dict[int, Counter[int]] = {0: Counter(), 1: Counter()}
    leak_counts: Counter[str] = Counter()
    seen: set[tuple[str, Any]] = set()
    usable = 0

    if len(records) != expected_total:
        errors.append(f"expected {expected_total} records, found {len(records)}")

    for record in records:
        key = (str(record.get("dataset")), record.get("index"))
        if key in seen:
            errors.append(f"duplicate record {key!r}")
            continue
        seen.add(key)

        label = record.get("label")
        rating = record.get("rating")
        structurally_usable = (
            not bool(record.get("parse_error"))
            and record.get("label_match") is True
            and record.get("rating_polarity_match") is True
            and label in (0, 1)
            and isinstance(rating, int)
            and 1 <= rating <= 7
        )
        if not structurally_usable:
            continue

        visible = "\n".join(
            str(record.get(field) or "")
            for field in ("reasoning_summary", "student_target")
        )
        leaked = False
        for name, pattern in LEAK_PATTERNS.items():
            if pattern.search(visible):
                leak_counts[name] += 1
                errors.append(f"{key!r}: visible target matches leakage rule {name}")
                leaked = True
        if leaked:
            continue

        usable += 1
        rating_counts[rating] += 1
        rating_counts_by_label[int(label)][rating] += 1

    distinct_ratings = len(rating_counts)
    intermediate = sum(rating_counts[rating] for rating in range(2, 7))
    intermediate_fraction = intermediate / usable if usable else 0.0
    label_zero_away_from_extreme = any(
        rating_counts_by_label[0][rating] for rating in (2, 3, 4)
    )
    label_one_away_from_extreme = any(
        rating_counts_by_label[1][rating] for rating in (4, 5, 6)
    )

    if usable < minimum_usable:
        errors.append(f"usable target gate failed: {usable} < {minimum_usable}")
    if distinct_ratings < minimum_distinct_ratings:
        errors.append(
            "rating diversity gate failed: "
            f"{distinct_ratings} < {minimum_distinct_ratings}"
        )
    if intermediate_fraction < minimum_intermediate_fraction:
        errors.append(
            "intermediate rating gate failed: "
            f"{intermediate_fraction:.4f} < {minimum_intermediate_fraction:.4f}"
        )
    if not label_zero_away_from_extreme:
        errors.append("label 0 has no usable rating away from extreme 1")
    if not label_one_away_from_extreme:
        errors.append("label 1 has no usable rating away from extreme 7")

    return {
        "records": len(records),
        "unique_records": len(seen),
        "usable": usable,
        "rating_counts": {
            str(rating): rating_counts[rating] for rating in range(1, 8)
        },
        "rating_counts_by_label": {
            str(label): {
                str(rating): rating_counts_by_label[label][rating]
                for rating in range(1, 8)
            }
            for label in (0, 1)
        },
        "distinct_ratings": distinct_ratings,
        "rating_entropy_bits": rating_entropy(rating_counts),
        "intermediate_count": intermediate,
        "intermediate_fraction": intermediate_fraction,
        "label_zero_away_from_extreme": label_zero_away_from_extreme,
        "label_one_away_from_extreme": label_one_away_from_extreme,
        "leak_counts": dict(sorted(leak_counts.items())),
        "errors": errors,
        "passed": not errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-total", required=True, type=int)
    parser.add_argument("--minimum-usable", required=True, type=int)
    parser.add_argument("--minimum-distinct-ratings", type=int, default=4)
    parser.add_argument("--minimum-intermediate-fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_records(
        load_records(args.artifact),
        expected_total=args.expected_total,
        minimum_usable=args.minimum_usable,
        minimum_distinct_ratings=args.minimum_distinct_ratings,
        minimum_intermediate_fraction=args.minimum_intermediate_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError(f"evidence-rating audit failed; see {args.output}")


if __name__ == "__main__":
    main()
