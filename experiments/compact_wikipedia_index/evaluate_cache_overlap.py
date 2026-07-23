#!/usr/bin/env python3
"""Report exact overlap between a generated cache and independent audited evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compact_wikipedia_index.core import source_matches


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--field", default="raw_real_passages")
    args = parser.parse_args()

    references = {
        (str(row["dataset"]), row["index"]): (
            row.get("real_passages") or row.get("passages") or []
        )
        for row in read_jsonl(args.references)
    }
    rows = read_jsonl(args.cache)
    active = 0
    matched_rows = 0
    passages = 0
    matched_passages = 0
    for row in rows:
        candidates = row.get(args.field) or []
        if not candidates:
            continue
        active += 1
        gold = references.get((str(row["dataset"]), row["index"]), [])
        row_matched = False
        for candidate in candidates:
            passages += 1
            matched = any(source_matches(candidate, reference) for reference in gold)
            matched_passages += int(matched)
            row_matched |= matched
        matched_rows += int(row_matched)
    print(json.dumps({
        "field": args.field,
        "active_rows": active,
        "passages": passages,
        "exact_audited_rows": matched_rows,
        "exact_audited_row_rate": matched_rows / active if active else 0.0,
        "exact_audited_passages": matched_passages,
        "exact_audited_passage_rate": matched_passages / passages if passages else 0.0,
    }, indent=2))


if __name__ == "__main__":
    main()
