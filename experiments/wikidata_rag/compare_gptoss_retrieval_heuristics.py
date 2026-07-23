#!/usr/bin/env python3
"""Compare conservative candidate selectors against GPT-OSS decisiveness labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable

from experiments.wikidata_rag.claim_retrieval import normalize, normalize_name


DECISIVE = {"decisive", "supports", "contradicts"}


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def subject_matches(row: dict[str, Any], candidate: dict[str, Any]) -> bool:
    subject = normalize_name(candidate["subject"])
    return bool(subject) and any(
        subject == normalize_name(span) for span in row.get("subjects", [])
    )


def routed(row: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return candidate["predicate"] in set(row.get("rule_predicates", []))


def value_in_answer(row: dict[str, Any], candidate: dict[str, Any]) -> bool:
    value = candidate["fact"].partition(":")[2].strip()
    return bool(normalize(value)) and normalize(value) in normalize(row["answer_full"])


def first_matching(
    row: dict[str, Any], predicate: Callable[[dict[str, Any], dict[str, Any]], bool]
) -> dict[str, Any] | None:
    return next((candidate for candidate in row["candidates"] if predicate(row, candidate)), None)


def evaluate(
    rows: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> dict[str, float]:
    emitted = correct = rows_with_decisive = 0
    for row in rows:
        decisive_ids = {
            item["id"] for item in row["labels"] if item["label"] in DECISIVE
        }
        rows_with_decisive += int(bool(decisive_ids))
        selected = selector(row)
        if selected is not None:
            emitted += 1
            correct += int(selected["id"] in decisive_ids)
    return {
        "rows": len(rows), "rows_with_decisive": rows_with_decisive,
        "emitted": emitted, "decisive": correct,
        "precision": correct / max(1, emitted),
        "recall_of_decisive_rows": correct / max(1, rows_with_decisive),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [row for row in load(args.input) if not row.get("parse_error")]
    selectors = {
        "candidate_order_top1": lambda row: row["candidates"][0] if row["candidates"] else None,
        "routed_subject": lambda row: first_matching(
            row, lambda source, candidate: routed(source, candidate) and subject_matches(source, candidate)
        ),
        "routed": lambda row: first_matching(row, routed),
        "answer_value": lambda row: first_matching(row, value_in_answer),
        "routed_subject_answer_value": lambda row: first_matching(
            row, lambda source, candidate: (
                routed(source, candidate)
                and subject_matches(source, candidate)
                and value_in_answer(source, candidate)
            )
        ),
    }
    report = {
        "rows": len(rows),
        "labels": Counter(item["label"] for row in rows for item in row["labels"]),
        "selectors": {name: evaluate(rows, selector) for name, selector in selectors.items()},
    }
    text = json.dumps(report, indent=2, default=dict) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
