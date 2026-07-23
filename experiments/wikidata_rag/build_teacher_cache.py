#!/usr/bin/env python3
"""Convert frozen offline Wikidata retrievals into teacher/student cache rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.wikidata_rag.build_validation_cache import entity_passages


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    entities = record.get("entities") or []
    passages = record.get("real_passages") or entity_passages(entities)
    return {
        "dataset": record["dataset"],
        "index": record["index"],
        "passages": passages,
        "qids": record.get("qids") or [entity.get("qid", "") for entity in entities],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=2880)
    args = parser.parse_args()

    records = [
        convert_record(json.loads(line))
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    keys = {(record["dataset"], record["index"]) for record in records}
    if len(keys) != len(records):
        raise RuntimeError("retrieval input contains duplicate dataset/index keys")
    if args.expected_rows and len(records) != args.expected_rows:
        raise RuntimeError(
            f"expected {args.expected_rows} rows, found {len(records)}"
        )
    if any(not record["passages"] for record in records):
        raise RuntimeError("one or more retrieval rows contain no passages")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )
    print(
        f"wrote {len(records)} rows and "
        f"{sum(len(record['passages']) for record in records)} passages to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
