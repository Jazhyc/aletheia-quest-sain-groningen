#!/usr/bin/env python3
"""Select claim evidence from cached full Wikipedia pages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.core import (
    deduplicate_passages,
    lexical_relevance,
    select_document_sentences,
    split_sentences,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_window_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not sizes or any(size < 1 or size > 3 for size in sizes):
        raise argparse.ArgumentTypeError(
            "window sizes must be comma-separated integers in [1, 3]"
        )
    return sizes


def select_document_spans(
    query: str,
    sentences: list[str],
    *,
    limit: int,
    window_sizes: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Select a quota-balanced mixture of sentences and adjacent windows."""
    if window_sizes == (1,):
        return [
            {
                "sentence_index": index,
                "sentence_end_index": index,
                "window_size": 1,
                "text": sentence,
                "lexical_score": score,
            }
            for index, sentence, score in select_document_sentences(
                query, sentences, limit=limit
            )
        ]
    candidates: dict[int, list[dict[str, Any]]] = {}
    for size in window_sizes:
        items = []
        for start in range(max(0, len(sentences) - size + 1)):
            text = " ".join(sentences[start : start + size])
            items.append({
                "sentence_index": start,
                "sentence_end_index": start + size - 1,
                "window_size": size,
                "text": text,
                "lexical_score": lexical_relevance(query, text),
            })
        candidates[size] = items
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    quota = math.ceil(limit / len(window_sizes))
    for size in window_sizes:
        ranked = sorted(
            candidates[size],
            key=lambda item: (
                float(item["lexical_score"]),
                -int(item["sentence_index"]),
            ),
            reverse=True,
        )
        if size == 1:
            # Preserve lead context even when its lexical overlap is low.
            lead = candidates[size][: min(2, quota)]
            ranked = lead + [item for item in ranked if item not in lead]
        for item in ranked[:quota]:
            selected[(int(item["sentence_index"]), int(item["window_size"]))] = item
    if len(selected) < limit:
        remaining = [
            item
            for items in candidates.values()
            for item in items
            if (int(item["sentence_index"]), int(item["window_size"])) not in selected
        ]
        remaining.sort(
            key=lambda item: (
                float(item["lexical_score"]) - 0.02 * (int(item["window_size"]) - 1),
                -int(item["sentence_index"]),
            ),
            reverse=True,
        )
        for item in remaining:
            selected[(int(item["sentence_index"]), int(item["window_size"]))] = item
            if len(selected) >= limit:
                break
    return list(selected.values())[:limit]


def select_claim_passages(
    claim: dict[str, Any],
    page_cache: dict[str, dict[str, Any]],
    *,
    sentences_per_page: int,
    max_page_rank: int | None = None,
    query_source: str = "question_titles_full_page",
    window_sizes: tuple[int, ...] = (1,),
) -> tuple[list[dict[str, Any]], list[str]]:
    passages = []
    errors = []
    seen_titles = set()
    for original in claim.get("passages") or []:
        if (
            max_page_rank is not None
            and int(original.get("page_rank", 0)) > max_page_rank
        ):
            continue
        requested_title = str(original.get("title") or "")
        if not requested_title or requested_title in seen_titles:
            continue
        seen_titles.add(requested_title)
        cached = page_cache.get(requested_title)
        if cached is None:
            errors.append(f"missing page cache: {requested_title}")
            continue
        if cached.get("error"):
            errors.append(f"{requested_title}: {cached['error']}")
            continue
        sentences = split_sentences(str(cached.get("extract") or ""))
        selected = select_document_spans(
            str(claim["proposition"]),
            sentences,
            limit=sentences_per_page,
            window_sizes=window_sizes,
        )
        for span in selected:
            passages.append({
                "title": str(cached.get("canonical_title") or requested_title),
                "text": str(span["text"]),
                "url": str(cached.get("url") or ""),
                "page_rank": int(original.get("page_rank", 0)),
                "sentence_index": int(span["sentence_index"]),
                "sentence_end_index": int(span["sentence_end_index"]),
                "window_size": int(span["window_size"]),
                "hop": 0,
                "links": [],
                "lexical_score": float(span["lexical_score"]),
                "query_source": query_source,
            })
    passages = deduplicate_passages(passages)
    passages.sort(
        key=lambda item: (
            float(item.get("lexical_score", 0.0)) - 0.04 * int(item["page_rank"]),
            -int(item["sentence_index"]),
        ),
        reverse=True,
    )
    return passages, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--page-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sentences-per-page", type=int, default=24)
    parser.add_argument("--max-page-rank", type=int)
    parser.add_argument("--query-source", default="question_titles_full_page")
    parser.add_argument("--window-sizes", type=parse_window_sizes, default=(1,))
    args = parser.parse_args()

    claims = read_jsonl(args.retrieval)
    page_rows = read_jsonl(args.page_cache)
    page_cache = {str(row["requested_title"]): row for row in page_rows}
    output = []
    for claim in claims:
        passages, errors = select_claim_passages(
            claim,
            page_cache,
            sentences_per_page=args.sentences_per_page,
            max_page_rank=args.max_page_rank,
            query_source=args.query_source,
            window_sizes=args.window_sizes,
        )
        output.append({
            **{field: claim[field] for field in (
                "dataset", "index", "claim_index", "label", "quote",
                "proposition", "question", "teacher_assessment",
            )},
            "passages": passages,
            "retrieval_seconds": 0.0,
            "error": "; ".join(errors) or None,
            "retrieval_config": {
                "source": args.query_source,
                "sentences_per_page": args.sentences_per_page,
                "max_page_rank": args.max_page_rank,
                "window_sizes": args.window_sizes,
            },
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
