#!/usr/bin/env python3
"""Build deployable raw-text and atomic-query validation retrieval caches."""

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

from experiments.compact_wikipedia_index.core import retrieve
from experiments.fever_fact_verification.core import (
    last_user_message,
    normalize_text,
    split_sentences,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def record_output(record: dict[str, Any]) -> str:
    """Recover the final output from either the explicit field or cached prompt."""
    output = str(record.get("output") or "")
    if not output:
        prompt = str(record.get("prompt") or "")
        start = prompt.rfind("<output>")
        end = prompt.find("</output>", start + len("<output>"))
        output = (
            prompt[start + len("<output>") : end].strip()
            if start >= 0 and end >= 0
            else ""
        )
    return output


def raw_queries(record: dict[str, Any], *, maximum: int = 6) -> list[str]:
    """Use only inference-visible output sentences; no teacher assessment is used."""
    return split_sentences(record_output(record), minimum_chars=12)[:maximum]


def atomic_queries(record: dict[str, Any], *, maximum: int = 6) -> list[str]:
    """Return grounded teacher propositions as a non-deployable extraction ceiling."""
    output = record_output(record)
    queries = []
    for claim in record.get("claims") or []:
        quote = str(claim.get("quote") or "")
        proposition = normalize_text(str(claim.get("proposition") or ""))
        if quote and quote in output and proposition:
            queries.append(proposition)
        if len(queries) >= maximum:
            break
    return queries


def retrieve_row(
    connection: sqlite3.Connection,
    question: str,
    queries: list[str],
    *,
    threshold: float,
    maximum_passages: int,
) -> list[dict[str, Any]]:
    candidates = []
    for claim_index, query in enumerate(queries):
        ranked = retrieve(
            connection,
            f"{question} {query}",
            limit=1,
            candidate_limit=30,
        )
        if not ranked or float(ranked[0]["retrieval_score"]) < threshold:
            continue
        candidate = ranked[0]
        candidates.append({
            "title": candidate["title"],
            "text": f"Claim: {query}\nSource sentence: {candidate['text']}",
            "url": candidate["url"],
            "claim_index": claim_index,
            "retrieval_score": candidate["retrieval_score"],
            "lexical_score": candidate["lexical_score"],
            "title_score": candidate["title_score"],
            "number_recall": candidate["number_recall"],
            "passage_id": candidate["passage_id"],
        })
    candidates.sort(
        key=lambda row: (float(row["retrieval_score"]), -int(row["claim_index"])),
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


def add_matched_shuffle(
    rows: list[dict[str, Any]],
    *,
    real_field: str,
    shuffled_field: str,
) -> None:
    """Rotate passages within count buckets while requiring another dataset unit."""
    buckets: dict[int, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        buckets[len(row[real_field])].append(position)
    for count, positions in buckets.items():
        if count == 0:
            for position in positions:
                rows[position][shuffled_field] = []
            continue
        for offset, position in enumerate(positions):
            recipient = rows[position]
            donor = None
            for hop in range(1, len(positions) + 1):
                candidate = rows[positions[(offset + hop) % len(positions)]]
                if candidate["dataset"] != recipient["dataset"]:
                    donor = candidate
                    break
            if donor is None:
                donor_passages = []
                donor_keys = []
                for candidate in rows:
                    if candidate["dataset"] == recipient["dataset"]:
                        continue
                    for passage in candidate[real_field]:
                        donor_passages.append(dict(passage))
                        donor_keys.append({
                            "dataset": candidate["dataset"],
                            "index": candidate["index"],
                        })
                        if len(donor_passages) >= count:
                            break
                    if len(donor_passages) >= count:
                        break
                if len(donor_passages) < count:
                    raise RuntimeError(
                        f"cannot find cross-dataset shuffle passages for count={count}"
                    )
                recipient[shuffled_field] = donor_passages
                recipient[f"{shuffled_field}_donor"] = donor_keys
            else:
                recipient[shuffled_field] = [dict(item) for item in donor[real_field]]
                recipient[f"{shuffled_field}_donor"] = {
                    "dataset": donor["dataset"],
                    "index": donor["index"],
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--maximum-passages", type=int, default=2)
    args = parser.parse_args()

    source = [
        row
        for row in read_jsonl(args.records)
        if row.get("variant") == "material_assessed"
        and "varied-deception" in str(row.get("dataset") or "")
    ]
    keys = [(str(row["dataset"]), row["index"]) for row in source]
    if len(keys) != len(set(keys)):
        raise RuntimeError("material_assessed records contain duplicate dataset/index keys")
    rows = []
    with sqlite3.connect(f"file:{args.index}?mode=ro", uri=True) as connection:
        for record in source:
            question = last_user_message(str(record.get("prompt") or ""))
            raw = retrieve_row(
                connection,
                question,
                raw_queries(record),
                threshold=args.threshold,
                maximum_passages=args.maximum_passages,
            )
            atomic = retrieve_row(
                connection,
                question,
                atomic_queries(record),
                threshold=args.threshold,
                maximum_passages=args.maximum_passages,
            )
            rows.append({
                "dataset": str(record["dataset"]),
                "index": record["index"],
                "raw_real_passages": raw,
                "atomic_real_passages": atomic,
                "retrieval_config": {
                    "threshold": args.threshold,
                    "maximum_passages": args.maximum_passages,
                    "corpus": "train-question Wikipedia pages only",
                },
            })
    add_matched_shuffle(
        rows,
        real_field="raw_real_passages",
        shuffled_field="raw_shuffled_passages",
    )
    add_matched_shuffle(
        rows,
        real_field="atomic_real_passages",
        shuffled_field="atomic_shuffled_passages",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    print(json.dumps({
        "rows": len(rows),
        "raw_active_rows": sum(bool(row["raw_real_passages"]) for row in rows),
        "raw_passages": sum(len(row["raw_real_passages"]) for row in rows),
        "atomic_active_rows": sum(bool(row["atomic_real_passages"]) for row in rows),
        "atomic_passages": sum(len(row["atomic_real_passages"]) for row in rows),
        "threshold": args.threshold,
    }, indent=2))


if __name__ == "__main__":
    main()
