#!/usr/bin/env python3
"""Retrieve bounded Wikipedia windows for model-extracted propositions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compact_wikipedia_index.build_validation_cache import add_matched_shuffle
from experiments.compact_wikipedia_index.core import (
    LinearReranker,
    adjacent_window,
    retrieve,
    retrieve_proposition,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def retrieve_row(
    connection: sqlite3.Connection,
    row: dict[str, Any],
    reranker: LinearReranker | None,
    *,
    threshold: float,
    maximum_passages: int,
) -> list[dict[str, Any]]:
    candidates = []
    for claim_index, claim in enumerate(row.get("claims") or []):
        ranked = (
            retrieve_proposition(
                connection,
                str(row["question"]),
                str(claim),
                limit=1,
                candidate_limit=30,
                reranker=reranker,
            )
            if reranker is not None
            else retrieve(
                connection,
                f"{row['question']} {claim}",
                limit=1,
                candidate_limit=30,
            )
        )
        score_field = "reranker_score" if reranker is not None else "retrieval_score"
        if not ranked or float(ranked[0][score_field]) < threshold:
            continue
        candidate = ranked[0]
        source = adjacent_window(connection, candidate, radius=1)
        candidates.append({
            "title": candidate["title"],
            "text": f"Claim: {claim}\nSource sentence: {source}",
            "url": candidate["url"],
            "claim_index": claim_index,
            "sentence_index": candidate["sentence_index"],
            "passage_id": candidate["passage_id"],
            "selection_score": candidate[score_field],
            "retrieval_score": candidate["retrieval_score"],
        })
    candidates.sort(
        key=lambda candidate: (
            float(candidate["selection_score"]),
            -int(candidate["claim_index"]),
        ),
        reverse=True,
    )
    output = []
    seen: set[int] = set()
    for candidate in candidates:
        passage_id = int(candidate["passage_id"])
        if passage_id in seen:
            continue
        seen.add(passage_id)
        output.append(candidate)
        if len(output) >= maximum_passages:
            break
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--reranker", type=Path)
    parser.add_argument("--threshold", type=float, default=1.285)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-passages", type=int, default=2)
    args = parser.parse_args()

    reranker = LinearReranker.from_json(args.reranker) if args.reranker else None
    threshold = reranker.threshold if reranker is not None else args.threshold
    rows = read_jsonl(args.queries)
    output = []
    with sqlite3.connect(f"file:{args.index}?mode=ro", uri=True) as connection:
        for row in rows:
            passages = retrieve_row(
                connection,
                row,
                reranker,
                threshold=threshold,
                maximum_passages=args.maximum_passages,
            )
            output.append({
                "dataset": str(row["dataset"]),
                "index": row["index"],
                "real_passages": passages,
                "extractor_parse_valid": bool(row.get("parse_valid")),
                "retrieval_config": {
                    "query": "base-Qwen standalone claims",
                    "threshold": threshold,
                    "reranker": str(args.reranker) if args.reranker else None,
                    "window_radius": 1,
                    "maximum_passages": args.maximum_passages,
                },
            })
    add_matched_shuffle(
        output,
        real_field="real_passages",
        shuffled_field="shuffled_passages",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )
    print(json.dumps({
        "rows": len(output),
        "active_rows": sum(bool(row["real_passages"]) for row in output),
        "passages": sum(len(row["real_passages"]) for row in output),
        "threshold": threshold,
    }, indent=2))


if __name__ == "__main__":
    main()
