#!/usr/bin/env python3
"""Summarize coverage changes between two frozen claim-gate caches."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["dataset"]), int(row["index"])


def covered(row: dict[str, Any]) -> bool:
    return bool(row.get("real_passages"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expanded_rows = load(args.expanded)
    expanded = {key(row): row for row in expanded_rows}
    baseline = {key(row): row for row in load(args.baseline)} if args.baseline else {}
    expanded_keys = {row_key for row_key, row in expanded.items() if covered(row)}
    baseline_keys = {row_key for row_key, row in baseline.items() if covered(row)}
    report = {
        "rows": len(expanded),
        "expanded_covered": len(expanded_keys),
        "expanded_coverage": len(expanded_keys) / max(1, len(expanded)),
        "baseline_covered": len(baseline_keys) if baseline else None,
        "newly_covered": len(expanded_keys - baseline_keys) if baseline else None,
        "lost_coverage": len(baseline_keys - expanded_keys) if baseline else None,
        "abstain_reasons": Counter(str(row.get("abstain_reason")) for row in expanded_rows),
        "covered_relations": Counter(
            relation
            for row in expanded_rows if covered(row)
            for relation in row.get("claim", {}).get("relations", [])
        ),
        "covered_predicates": Counter(
            predicate
            for row in expanded_rows if covered(row)
            for entity in row.get("entities", [])
            for predicate in entity.get("gate", {}).get("predicates", [])
        ),
        "newly_covered_examples": [
            {
                "dataset": expanded[row_key]["dataset"],
                "index": expanded[row_key]["index"],
                "question": expanded[row_key].get("claim", {}).get("question", ""),
                "passages": expanded[row_key].get("real_passages", []),
            }
            for row_key in sorted(expanded_keys - baseline_keys)[:50]
        ] if baseline else [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
