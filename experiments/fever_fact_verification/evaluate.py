#!/usr/bin/env python3
"""Evaluate evidence specificity and noisy claim-verdict agreement."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ASSESSMENT_TO_VERDICT = {
    "true": "SUPPORTS",
    "false": "REFUTES",
    "uncertain": "NOT_ENOUGH_INFO",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    report: dict[str, Any] = {"rows": len(rows), "conditions": {}}
    for condition, items in sorted(by_condition.items()):
        assessable = [
            row for row in items
            if str(row["teacher_assessment"]).lower() in ASSESSMENT_TO_VERDICT
        ]
        matches = sum(
            row["verdict"] == ASSESSMENT_TO_VERDICT[str(row["teacher_assessment"]).lower()]
            for row in assessable
        )
        polar = [row for row in items if row["verdict"] != "NOT_ENOUGH_INFO"]
        polar_assessable = [
            row for row in polar
            if str(row["teacher_assessment"]).lower() in {"true", "false"}
        ]
        polar_matches = sum(
            row["verdict"] == ASSESSMENT_TO_VERDICT[str(row["teacher_assessment"]).lower()]
            for row in polar_assessable
        )
        report["conditions"][condition] = {
            "claims": len(items),
            "retrieval_errors": sum(bool(row.get("retrieval_error")) for row in items),
            "mean_candidates": (
                sum(int(row.get("candidate_count", 0)) for row in items) / len(items)
                if items else 0.0
            ),
            "verdicts": Counter(row["verdict"] for row in items),
            "coverage": len(polar) / len(items) if items else 0.0,
            "mean_confidence": (
                sum(float(row["confidence"]) for row in polar) / len(polar)
                if polar else 0.0
            ),
            "teacher_assessment_agreement": matches / len(assessable) if assessable else 0.0,
            "polar_teacher_precision": (
                polar_matches / len(polar_assessable) if polar_assessable else 0.0
            ),
            "verdict_by_teacher_assessment": {
                assessment: Counter(
                    row["verdict"]
                    for row in items
                    if str(row["teacher_assessment"]).lower() == assessment
                )
                for assessment in ("true", "false", "uncertain")
            },
            "polar_by_row_label": {
                str(label): Counter(
                    row["verdict"] for row in items if int(row["label"]) == label
                )
                for label in (0, 1)
            },
        }

    real = {
        (row["dataset"], row["index"], row["claim_index"]): row
        for row in by_condition.get("real", [])
    }
    shuffled = {
        (row["dataset"], row["index"], row["claim_index"]): row
        for row in by_condition.get("shuffled", [])
    }
    common = sorted(set(real) & set(shuffled))
    if common:
        report["paired_real_vs_shuffled"] = {
            "claims": len(common),
            "higher_non_neutral_confidence": sum(
                float(real[key]["confidence"]) > float(shuffled[key]["confidence"])
                for key in common
            ),
            "real_only_polar": sum(
                real[key]["verdict"] != "NOT_ENOUGH_INFO"
                and shuffled[key]["verdict"] == "NOT_ENOUGH_INFO"
                for key in common
            ),
            "shuffled_only_polar": sum(
                real[key]["verdict"] == "NOT_ENOUGH_INFO"
                and shuffled[key]["verdict"] != "NOT_ENOUGH_INFO"
                for key in common
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(read_jsonl(args.input))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
