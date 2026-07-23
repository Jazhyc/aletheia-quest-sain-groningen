#!/usr/bin/env python3
"""Summarize selective evidence audits against noisy claim assessments."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "true": "DECISIVE_SUPPORT",
    "false": "DECISIVE_CONTRADICTION",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisive = [row for row in rows if row.get("decision") != "ABSTAIN"]
    assessable = [
        row for row in decisive
        if str(row.get("teacher_assessment") or "").lower() in EXPECTED
    ]
    matches = sum(
        row["decision"] == EXPECTED[str(row["teacher_assessment"]).lower()]
        for row in assessable
    )
    candidate_relations = Counter(
        str(item.get("relation") or "MISSING")
        for row in rows
        for item in row.get("candidate_assessments") or []
    )
    return {
        "claims": len(rows),
        "decisions": Counter(str(row.get("decision")) for row in rows),
        "coverage": len(decisive) / len(rows) if rows else 0.0,
        "parse_errors": sum(not row.get("parse_valid") for row in rows),
        "candidate_relations": candidate_relations,
        "conditional_teacher_precision": matches / len(assessable) if assessable else 0.0,
        "decision_by_teacher_assessment": {
            assessment: Counter(
                str(row.get("decision"))
                for row in rows
                if str(row.get("teacher_assessment") or "").lower() == assessment
            )
            for assessment in ("true", "false", "uncertain")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = {
        path.stem: summarize(read_jsonl(path))
        for path in args.input
    }
    rendered = json.dumps(reports, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
