#!/usr/bin/env python3
"""Apply a frozen direct-score threshold to grounded Qwen planner selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.wikidata_rag.evaluate_qwen_retriever_rank1 import filtered_plans
from experiments.wikidata_rag.qwen_database_planner import evaluate_plans


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_map(
    rows: list[dict[str, Any]],
    score_field: str,
) -> dict[tuple[str, int, str], float]:
    """Index candidate scores from a frozen evaluation artifact."""
    return {
        (
            str(row["dataset"]),
            int(row["index"]),
            str(row["candidate_id"]),
        ): float(row[score_field])
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--score-field", default="base_score")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--teacher-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    plans = filtered_plans(
        load_jsonl(args.planner),
        score_map(load_jsonl(args.candidate_scores), args.score_field),
        args.threshold,
    )
    report = evaluate_plans(plans, load_jsonl(args.teacher_labels))
    report.update({
        "score_field": args.score_field,
        "threshold": args.threshold,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in plans) + "\n"
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
