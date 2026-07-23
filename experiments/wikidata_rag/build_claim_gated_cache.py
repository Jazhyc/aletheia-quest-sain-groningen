#!/usr/bin/env python3
"""Build a frozen cache using rule-based claim-focused Wikidata retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.wikidata_rag.claim_retrieval import retrieve_claim_evidence


def entity_passages(entities: list[dict[str, object]]) -> list[dict[str, str]]:
    """Render only the relation-aligned facts retained by the gate."""
    passages = []
    for entity in entities:
        facts = "; ".join(str(fact) for fact in entity.get("facts", []))
        passages.append({
            "title": f"{entity.get('label', '')} ({entity.get('qid', '')})",
            "text": facts,
        })
    return passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-results", type=int, default=1)
    args = parser.parse_args()

    source = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    rows = []
    for number, record in enumerate(source, 1):
        result = retrieve_claim_evidence(
            connection, record["conversation"], limit=args.limit_results
        )
        entities = result["entities"]
        passages = entity_passages(entities)
        rows.append({
            "dataset": record["dataset"],
            "index": record["index"],
            "conversation": record["conversation"],
            "claim": result["claim"],
            "real_passages": passages,
            "passages": passages,
            "entities": entities,
            "qids": [entity["qid"] for entity in entities],
            "abstain_reason": result["abstain_reason"],
        })
        if number % 100 == 0:
            print(f"processed {number}/{len(source)}", flush=True)
    connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    covered = sum(bool(row["real_passages"]) for row in rows)
    print(f"wrote {len(rows)} rows; covered={covered}; abstained={len(rows) - covered}")


if __name__ == "__main__":
    main()
