#!/usr/bin/env python3
"""Audit Liars' Bench privileged summaries before student training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any


LEAK_PATTERNS = {
    "ground_truth_label": re.compile(r"\bground[- ]truth label\b", re.IGNORECASE),
    "authoritative_label": re.compile(r"\bauthoritative label\b", re.IGNORECASE),
    "provided_label": re.compile(r"\b(?:provided|given) label\b", re.IGNORECASE),
    "privileged_evidence": re.compile(r"\bprivileged (?:evidence|information)\b", re.IGNORECASE),
    "teacher_only": re.compile(r"\bteacher[- ]only\b", re.IGNORECASE),
    "recorded_belief": re.compile(r"\brecorded (?:answer[- ]?)?belief\b", re.IGNORECASE),
    "metadata": re.compile(r"\bmetadata\b", re.IGNORECASE),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL artifact and reject blank/non-object records."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(record)
    return records


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return coverage, balance, length, and privileged-language diagnostics."""
    total = len(records)
    usable = [
        record
        for record in records
        if not record.get("parse_error")
        and record.get("label_match") is True
        and record.get("student_target")
        and record.get("reasoning_summary")
    ]
    label_counts = Counter(str(record.get("label")) for record in usable)
    dataset_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    word_counts: list[int] = []
    leak_counts: Counter[str] = Counter()
    leak_rows: list[dict[str, Any]] = []

    for record in usable:
        dataset = str(record.get("dataset"))
        label = str(record.get("label"))
        dataset_label_counts[dataset][label] += 1
        summary = str(record.get("reasoning_summary") or "")
        word_counts.append(len(summary.split()))
        matched = [name for name, pattern in LEAK_PATTERNS.items() if pattern.search(summary)]
        if matched:
            leak_counts.update(matched)
            leak_rows.append({
                "dataset": dataset,
                "index": str(record.get("index")),
                "patterns": matched,
                "summary": summary,
            })

    usable_count = len(usable)
    return {
        "total": total,
        "usable": usable_count,
        "coverage": usable_count / total if total else 0.0,
        "label_counts": dict(sorted(label_counts.items())),
        "dataset_label_counts": {
            dataset: dict(sorted(counts.items()))
            for dataset, counts in sorted(dataset_label_counts.items())
        },
        "summary_words": {
            "mean": sum(word_counts) / len(word_counts) if word_counts else 0.0,
            "max": max(word_counts, default=0),
        },
        "leak_rows": len(leak_rows),
        "leak_fraction": len(leak_rows) / usable_count if usable_count else 0.0,
        "leak_pattern_counts": dict(sorted(leak_counts.items())),
        "leak_examples": leak_rows[:10],
    }


def validate_audit(
    audit: dict[str, Any],
    *,
    min_coverage: float,
    max_leak_fraction: float,
    max_label_imbalance_fraction: float = 0.05,
) -> list[str]:
    """Return human-readable quality-gate failures."""
    failures: list[str] = []
    if audit["coverage"] < min_coverage:
        failures.append(
            f"coverage {audit['coverage']:.3f} is below required {min_coverage:.3f}"
        )
    if audit["leak_fraction"] > max_leak_fraction:
        failures.append(
            "privileged-language leak fraction "
            f"{audit['leak_fraction']:.3f} exceeds allowed {max_leak_fraction:.3f}"
        )
    for dataset, counts in audit["dataset_label_counts"].items():
        negative = counts.get("0", 0)
        positive = counts.get("1", 0)
        denominator = negative + positive
        imbalance = abs(negative - positive) / denominator if denominator else 1.0
        if imbalance > max_label_imbalance_fraction:
            failures.append(
                f"usable label imbalance {imbalance:.3f} for {dataset} exceeds "
                f"allowed {max_label_imbalance_fraction:.3f}: {counts}"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--max-leak-fraction", type=float, default=0.05)
    parser.add_argument("--max-label-imbalance-fraction", type=float, default=0.05)
    args = parser.parse_args()

    audit = audit_records(load_jsonl(args.artifact))
    rendered = json.dumps(audit, indent=2, ensure_ascii=False)
    print(rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")

    failures = validate_audit(
        audit,
        min_coverage=args.min_coverage,
        max_leak_fraction=args.max_leak_fraction,
        max_label_imbalance_fraction=args.max_label_imbalance_fraction,
    )
    if failures:
        raise SystemExit("teacher-cache audit failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
