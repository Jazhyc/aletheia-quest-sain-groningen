#!/usr/bin/env python3
"""Summarize and expose audit examples from GPT-OSS retrieval labels."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from typing import Any


DECISIVE = {"supports", "contradicts", "decisive"}


def duplicate_question_consistency(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[dict[tuple[str, str], str]]] = {}
    for row in rows:
        candidates = {item["id"]: item for item in row["candidates"]}
        labels = {
            (str(candidates[item["id"]]["qid"]), candidates[item["id"]]["fact"]): item["label"]
            for item in row["labels"] if item["id"] in candidates
        }
        grouped.setdefault(row["question_group"], []).append(labels)
    comparisons = exact = binary = decisive_conflicts = 0
    for group_rows in grouped.values():
        for left, right in combinations(group_rows, 2):
            for key in left.keys() & right.keys():
                comparisons += 1
                exact += int(left[key] == right[key])
                left_decisive = left[key] in DECISIVE
                right_decisive = right[key] in DECISIVE
                binary += int(left_decisive == right_decisive)
                decisive_conflicts += int(left_decisive != right_decisive)
    return {
        "overlapping_candidate_pairs": comparisons,
        "exact_label_agreement": exact / max(1, comparisons),
        "decisive_binary_agreement": binary / max(1, comparisons),
        "decisive_conflicts": decisive_conflicts,
    }


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()

    rows = load(args.input)
    valid = [row for row in rows if not row.get("parse_error")]
    labels = Counter(item["label"] for row in valid for item in row["labels"])
    decisive_rows = [
        row for row in valid
        if any(item["label"] in DECISIVE for item in row["labels"])
    ]
    report = {
        "rows": len(rows), "valid_rows": len(valid),
        "parse_errors": len(rows) - len(valid), "labels": labels,
        "rows_with_decisive_fact": len(decisive_rows),
        "decisive_row_rate": len(decisive_rows) / max(1, len(valid)),
        "coverage_cross_tab": {
            "current_covered_rows": sum(bool(row.get("currently_covered")) for row in valid),
            "current_covered_with_decisive": sum(
                bool(row.get("currently_covered")) for row in decisive_rows
            ),
            "currently_uncovered_with_decisive": sum(
                not bool(row.get("currently_covered")) for row in decisive_rows
            ),
        },
        "duplicate_question_consistency": duplicate_question_consistency(valid),
        "validation_errors": Counter(
            error.partition(":")[0]
            for row in rows for error in row.get("validation_errors", [])
        ),
        "audit_examples": [],
    }
    for row in decisive_rows[:args.examples]:
        candidates = {item["id"]: item for item in row["candidates"]}
        report["audit_examples"].append({
            "dataset": row["dataset"], "index": row["index"],
            "question": row["question"], "answer": row["answer_full"],
            "decisive": [
                {**item, "candidate": candidates[item["id"]]}
                for item in row["labels"] if item["label"] in DECISIVE
            ],
        })
    text = json.dumps(report, indent=2, ensure_ascii=False, default=dict) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
