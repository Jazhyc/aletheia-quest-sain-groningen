#!/usr/bin/env python3
"""Build balanced real/counterfactual polarity pairs for Qwen LoRA SFT."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from experiments.wikidata_rag.build_qwen_retriever_distillation import group_bucket


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def polarity_prompt(row: dict[str, Any], candidate: dict[str, Any]) -> str:
    return f"""You are checking a direct factual answer against one database fact.

The candidate has already passed a separate relevance check. Predict 1 only
when the database evidence contradicts the response's direct answer. Predict 0
when it supports that answer. Compare the answer values; do not infer intent or
deception.

Question:
{row["question"]}

Response:
{str(row["answer_full"])[:2400]}

Database evidence:
Subject: {candidate["subject"]}
Statement: {candidate["fact"]}

Output exactly one binary decision:
"""


def replace_database_value(
    candidate: dict[str, Any],
    old_value: str,
    new_value: str,
) -> dict[str, Any] | None:
    """Return a support twin by replacing a grounded database value."""
    subject = str(candidate["subject"])
    fact = str(candidate["fact"])
    replaced_subject = subject.replace(old_value, new_value)
    replaced_fact = fact.replace(old_value, new_value)
    if (replaced_subject, replaced_fact) == (subject, fact):
        return None
    return {
        **candidate,
        "subject": replaced_subject,
        "fact": replaced_fact,
    }


def training_record(
    row: dict[str, Any],
    candidate: dict[str, Any],
    *,
    pair_id: str,
    label: int,
    variant: str,
) -> dict[str, Any]:
    return {
        "dataset": pair_id,
        "index": variant,
        "label": label,
        "label_match": True,
        "parse_error": False,
        "student_prompt": polarity_prompt(row, candidate),
        "student_target": f"Prediction:{label}",
        "source_row_index": row["source_row_index"],
        "question_group": row["question_group"],
        "candidate_id": row["candidate"]["id"],
        "polarity_variant": variant,
    }


def build_pairs(
    rows: list[dict[str, Any]],
    *,
    fit_buckets: set[int] = {2, 3, 4},
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair each real contradiction with a value-matched support twin."""
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    groups: set[str] = set()
    for row in rows:
        bucket = group_bucket(str(row["question_group"]))
        counts[f"source_bucket_{bucket}"] += 1
        if bucket not in fit_buckets:
            continue
        annotation = row.get("annotation") or {}
        polarity = annotation.get("polarity")
        if polarity != "contradicts":
            counts[f"skipped_{polarity or 'abstain'}"] += 1
            continue
        twin = replace_database_value(
            row["candidate"],
            str(annotation["database_value"]),
            str(annotation["claimed_value"]),
        )
        if twin is None:
            counts["skipped_unreplaceable"] += 1
            continue
        pair_id = (
            f"polarity-pair:{row['source_row_index']}:{row['candidate']['id']}"
        )
        records.extend(
            [
                training_record(
                    row,
                    row["candidate"],
                    pair_id=pair_id,
                    label=1,
                    variant="real-contradiction",
                ),
                training_record(
                    row,
                    twin,
                    pair_id=pair_id,
                    label=0,
                    variant="counterfactual-support",
                ),
            ]
        )
        groups.add(str(row["question_group"]))
        counts["pairs"] += 1
    return records, {
        "records": len(records),
        "pairs": counts["pairs"],
        "question_groups": len(groups),
        "fit_buckets": sorted(fit_buckets),
        "counts": dict(sorted(counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-pairs", type=int, default=0)
    args = parser.parse_args()
    records, report = build_pairs(load(args.input))
    if report["pairs"] < args.min_pairs:
        raise RuntimeError(
            f"polarity gate failed: {report['pairs']} pairs < {args.min_pairs}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
