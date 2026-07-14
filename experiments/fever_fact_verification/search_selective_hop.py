#!/usr/bin/env python3
"""Search Wikipedia titles from label-blind abstention queries."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.core import (
    select_document_sentences,
    split_sentences,
)
from experiments.fever_fact_verification.retrieve_wikipedia import request_json
from experiments.fever_fact_verification.selective_hop import claim_key


TAG = re.compile(r"<[^>]+>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def search_titles(query: str, page_limit: int) -> list[dict[str, Any]]:
    payload = request_json({
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "list": "search",
        "srsearch": query,
        "srnamespace": 0,
        "srlimit": page_limit,
        "srprop": "snippet",
        "maxlag": 5,
    })
    return [
        {
            "title": str(item.get("title") or ""),
            "text": " ".join(html.unescape(TAG.sub(" ", str(item.get("snippet") or ""))).split()),
            "url": "",
            "page_rank": rank,
            "sentence_index": 0,
            "hop": 1,
            "links": [],
            "query_source": "gpt_oss_label_blind_abstention",
        }
        for rank, item in enumerate(payload.get("query", {}).get("search", []))
    ]


def search_full_page(
    query: str, proposition: str, *, max_sentences: int
) -> list[dict[str, Any]]:
    """Fetch the top search result and select proposition-relevant full-page text."""
    payload = request_json({
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 0,
        "gsrlimit": 1,
        "prop": "extracts|info",
        "explaintext": 1,
        "inprop": "url",
        "redirects": 1,
        "maxlag": 5,
    })
    pages = payload.get("query", {}).get("pages", [])
    passages = []
    for page in pages:
        sentences = split_sentences(str(page.get("extract") or ""))
        for sentence_index, sentence, lexical_score in select_document_sentences(
            proposition, sentences, limit=max_sentences
        ):
            passages.append({
                "title": str(page.get("title") or ""),
                "text": sentence,
                "url": str(page.get("fullurl") or ""),
                "page_rank": 0,
                "sentence_index": sentence_index,
                "hop": 1,
                "links": [],
                "lexical_score": lexical_score,
                "query_source": "gpt_oss_label_blind_abstention_full_page",
            })
    passages.sort(
        key=lambda item: (
            float(item.get("lexical_score", 0.0)),
            -int(item.get("sentence_index", 0)),
        ),
        reverse=True,
    )
    return passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-limit", type=int, default=2)
    parser.add_argument("--full-pages", action="store_true")
    parser.add_argument("--max-sentences", type=int, default=24)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sample-abstentions",
        type=int,
        help="Deterministically sample this many abstaining claims before retrieval.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.audit)
    if args.sample_abstentions is not None:
        abstentions = [row for row in rows if row.get("decision") == "ABSTAIN"]
        rng = random.Random(args.seed)
        rows = rng.sample(
            abstentions,
            k=min(args.sample_abstentions, len(abstentions)),
        )
        rows.sort(key=claim_key)
    if args.limit is not None:
        rows = rows[: args.limit]
    existing = read_jsonl(args.output) if args.output.exists() else []
    completed = {claim_key(row) for row in existing if not row.get("error")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.output.exists() else "w") as handle:
        for position, row in enumerate(rows, start=1):
            if claim_key(row) in completed:
                continue
            started = time.perf_counter()
            passages = []
            error = None
            if row.get("decision") == "ABSTAIN":
                try:
                    query = str(row.get("query") or row["proposition"])
                    passages = (
                        search_full_page(
                            query,
                            str(row["proposition"]),
                            max_sentences=args.max_sentences,
                        )
                        if args.full_pages
                        else search_titles(query, args.page_limit)
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            handle.write(json.dumps({
                **{field: row[field] for field in (
                    "dataset", "index", "claim_index", "label", "quote",
                    "proposition", "question", "teacher_assessment",
                ) if field in row},
                "selective_query": str(row.get("query") or ""),
                "passages": passages,
                "retrieval_seconds": time.perf_counter() - started,
                "error": error,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 50 == 0:
                print(json.dumps({"processed": position, "total": len(rows)}), flush=True)
            if row.get("decision") == "ABSTAIN":
                time.sleep(args.delay)


if __name__ == "__main__":
    main()
