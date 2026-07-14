#!/usr/bin/env python3
"""Select claim evidence from cached full Wikipedia pages."""

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
    select_document_sentences,
    split_sentences,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_claim_passages(
    claim: dict[str, Any],
    page_cache: dict[str, dict[str, Any]],
    *,
    sentences_per_page: int,
    max_page_rank: int | None = None,
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
        selected = select_document_sentences(
            str(claim["proposition"]),
            sentences,
            limit=sentences_per_page,
        )
        for sentence_index, sentence, lexical_score in selected:
            passages.append({
                "title": str(cached.get("canonical_title") or requested_title),
                "text": sentence,
                "url": str(cached.get("url") or ""),
                "page_rank": int(original.get("page_rank", 0)),
                "sentence_index": sentence_index,
                "hop": 0,
                "links": [],
                "lexical_score": lexical_score,
                "query_source": "question_titles_full_page",
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
                "source": "question_titles_full_page",
                "sentences_per_page": args.sentences_per_page,
                "max_page_rank": args.max_page_rank,
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
