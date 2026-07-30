#!/usr/bin/env python3
"""Audit completeness, quality, and cost of the full Kimi Liars cache."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--students", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=6576)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def auc(rows: list[dict]) -> float:
    return float(
        roc_auc_score(
            [int(row["label"]) for row in rows],
            [float(row["score"]) for row in rows],
        )
    )


def summarize(rows: list[dict]) -> dict:
    labels = Counter(int(row["label"]) for row in rows)
    scores = [float(row["score"]) for row in rows]
    return {
        "n": len(rows),
        "labels": {str(key): value for key, value in sorted(labels.items())},
        "auroc": auc(rows),
        "mean_score_by_label": {
            str(label): float(np.mean([
                float(row["score"]) for row in rows if int(row["label"]) == label
            ]))
            for label in (0, 1)
        },
        "unique_scores": len(set(scores)),
        "retry_rows": sum(int(row.get("attempts", 1)) > 1 for row in rows),
        "missing_binary_token_rows": sum(
            bool(row.get("missing_binary_tokens")) for row in rows
        ),
    }


def main() -> None:
    args = parse_args()
    generations = load_jsonl(args.generations)
    students = load_jsonl(args.students)
    if len(generations) != args.expected_rows:
        raise ValueError(
            f"expected {args.expected_rows} generations, got {len(generations)}"
        )
    generation_by_key = {
        (str(row["dataset"]), str(row["index"])): row for row in generations
    }
    student_by_key = {
        (str(row["dataset"]), str(row["index"])): row for row in students
    }
    if len(generation_by_key) != len(generations):
        raise ValueError("generation identities are not unique")
    if len(student_by_key) != len(students):
        raise ValueError("student identities are not unique")
    if set(generation_by_key) != set(student_by_key):
        raise ValueError("generation and student identities differ")
    for key, generation in generation_by_key.items():
        student = student_by_key[key]
        if int(generation["label"]) != int(student["label"]):
            raise ValueError(f"label mismatch for {key}")
        if str(generation.get("condition")) != str(student["teacher_prompt_kind"]):
            raise ValueError(f"teacher-condition mismatch for {key}")

    by_category: dict[str, list[dict]] = defaultdict(list)
    by_category_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for key, generation in generation_by_key.items():
        student = student_by_key[key]
        category = key[0].removeprefix("liars-bench/")
        source = str(student["source_model"])
        by_category[category].append(generation)
        by_category_source[(category, source)].append(generation)
    numeric_usage_keys = sorted({
        key
        for row in generations
        for key, value in (row.get("usage") or {}).items()
        if isinstance(value, (int, float))
    })
    result = {
        "overall": summarize(generations),
        "categories": {
            category: {
                **summarize(rows),
                "condition": str(rows[0]["condition"]),
            }
            for category, rows in sorted(by_category.items())
        },
        "category_sources": {
            f"{category}/{source}": summarize(rows)
            for (category, source), rows in sorted(by_category_source.items())
            if len({int(row["label"]) for row in rows}) == 2
        },
        "conditions": dict(Counter(
            str(row["condition"]) for row in generations
        )),
        "providers": dict(Counter(
            str(row.get("provider")) for row in generations
        )),
        "response_models": dict(Counter(
            str(row.get("model")) for row in generations
        )),
        "usage": {
            key: float(sum(
                float((row.get("usage") or {}).get(key, 0)) for row in generations
            ))
            for key in numeric_usage_keys
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

