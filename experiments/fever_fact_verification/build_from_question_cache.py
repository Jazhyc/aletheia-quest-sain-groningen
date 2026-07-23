#!/usr/bin/env python3
"""Join grounded claims to an independently retrieved question-level Wiki cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.core import (
    deduplicate_passages,
    load_grounded_claims,
    split_sentences,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_cached_passages(row: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for page_rank, passage in enumerate(row.get("passages") or []):
        title = str(passage.get("title") or "")
        sentences = split_sentences(str(passage.get("text") or ""), minimum_chars=12)
        for sentence_index, sentence in enumerate(sentences):
            output.append({
                "title": title,
                "text": sentence,
                "url": "",
                "page_rank": page_rank,
                "sentence_index": sentence_index,
                "hop": 0,
                "links": [],
                "query_source": "last_user_question",
            })
    return deduplicate_passages(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--question-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="material_assessed")
    parser.add_argument("--dataset-name-contains")
    args = parser.parse_args()

    claims = load_grounded_claims(
        read_jsonl(args.claims),
        variant=args.variant,
        dataset_name_contains=args.dataset_name_contains,
    )
    cache_rows = read_jsonl(args.question_cache)
    cache = {
        (str(row["dataset"]), row["index"]): row
        for row in cache_rows
    }
    if len(cache) != len(cache_rows):
        raise ValueError("duplicate dataset/index rows in question cache")
    output = []
    missing = 0
    query_mismatches = 0
    for claim in claims:
        key = (str(claim["dataset"]), claim["index"])
        cached = cache.get(key)
        if cached is None:
            missing += 1
            passages = []
        else:
            if claim["question"] and claim["question"] != str(cached.get("query") or ""):
                query_mismatches += 1
            passages = normalize_cached_passages(cached)
        output.append({
            **claim,
            "passages": passages,
            "retrieval_seconds": 0.0,
            "error": "missing_question_cache" if cached is None else cached.get("error"),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )
    print(json.dumps({
        "claims": len(output),
        "source_rows": len({(row["dataset"], row["index"]) for row in output}),
        "missing_cache": missing,
        "query_mismatches": query_mismatches,
        "passages": sum(len(row["passages"]) for row in output),
    }, indent=2))


if __name__ == "__main__":
    main()
