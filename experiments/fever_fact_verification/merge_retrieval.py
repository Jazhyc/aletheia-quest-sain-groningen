#!/usr/bin/env python3
"""Merge multiple claim retrieval channels without changing claim identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.core import deduplicate_passages


Key = tuple[str, Any, int]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row: dict[str, Any]) -> Key:
    return str(row["dataset"]), row["index"], int(row["claim_index"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [read_jsonl(path) for path in args.input]
    maps = [{key(row): row for row in rows} for rows in sources]
    if any(len(mapping) != len(rows) for mapping, rows in zip(maps, sources)):
        raise ValueError("duplicate claim key within a retrieval input")
    keys = set().union(*(mapping.keys() for mapping in maps))
    output = []
    for claim_key in sorted(keys):
        present = [mapping[claim_key] for mapping in maps if claim_key in mapping]
        base = present[0]
        identity = {
            (row["proposition"], row["quote"], row["teacher_assessment"])
            for row in present
        }
        if len(identity) != 1:
            raise ValueError(f"claim mismatch across retrieval inputs: {claim_key}")
        passages = []
        errors = []
        for source_index, row in enumerate(present):
            passages.extend(
                {**passage, "retrieval_channel": source_index}
                for passage in row.get("passages") or []
            )
            if row.get("error"):
                errors.append(str(row["error"]))
        output.append({
            **{field: base[field] for field in (
                "dataset", "index", "claim_index", "label", "quote",
                "proposition", "question", "teacher_assessment",
            )},
            "passages": deduplicate_passages(passages),
            "retrieval_seconds": sum(float(row.get("retrieval_seconds", 0.0)) for row in present),
            "error": "; ".join(errors) or None,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )
    print(json.dumps({
        "claims": len(output),
        "passages": sum(len(row["passages"]) for row in output),
        "errors": sum(bool(row["error"]) for row in output),
    }, indent=2))


if __name__ == "__main__":
    main()
