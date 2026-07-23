#!/usr/bin/env python3
"""Measure whether a frozen index recovers independently audited references."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compact_wikipedia_index.core import retrieve, source_matches


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    references = read_jsonl(args.references)
    claims = 0
    rows: set[tuple[str, Any]] = set()
    recovered_claims = defaultdict(int)
    recovered_rows: dict[int, set[tuple[str, Any]]] = defaultdict(set)
    details = []
    with sqlite3.connect(f"file:{args.index}?mode=ro", uri=True) as connection:
        for row in references:
            for reference in row.get("real_passages") or row.get("passages") or []:
                query = str(reference.get("text") or "").split("\n", 1)[0]
                query = query.removeprefix("Claim:").strip()
                if not query:
                    continue
                claims += 1
                row_key = (str(row["dataset"]), row["index"])
                rows.add(row_key)
                candidates = retrieve(connection, query, limit=args.top_k)
                rank = next(
                    (
                        position
                        for position, candidate in enumerate(candidates, start=1)
                        if source_matches(candidate, reference)
                    ),
                    None,
                )
                if rank is not None:
                    for cutoff in (1, 3, 5, 10):
                        if rank <= cutoff:
                            recovered_claims[cutoff] += 1
                            recovered_rows[cutoff].add(row_key)
                details.append({
                    "dataset": row["dataset"],
                    "index": row["index"],
                    "claim_index": reference.get("claim_index"),
                    "query": query,
                    "reference": reference,
                    "match_rank": rank,
                    "candidates": candidates,
                })
    report = {
        "reference_claims": claims,
        "reference_rows": len(rows),
        "recovery": {
            f"top_{cutoff}": {
                "claims": recovered_claims[cutoff],
                "claim_recall": recovered_claims[cutoff] / claims if claims else 0.0,
                "rows": len(recovered_rows[cutoff]),
                "row_recall": len(recovered_rows[cutoff]) / len(rows) if rows else 0.0,
            }
            for cutoff in (1, 3, 5, 10)
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
