#!/usr/bin/env python3
"""Retrieve sentence evidence for grounded claims from global English Wikipedia."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.core import (
    deduplicate_passages,
    load_grounded_claims,
    lexical_relevance,
    select_document_sentences,
    split_sentences,
)


API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "AletheiasQuestResearch/1.0 (offline fact-verification experiment)"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def request_json(
    params: dict[str, Any],
    *,
    retries: int = 5,
    timeout: float = 30.0,
) -> dict[str, Any]:
    url = f"{API_URL}?{urlencode(params)}"
    headers = {"User-Agent": USER_AGENT}
    access_token = os.environ.get("WIKIMEDIA_ACCESS_TOKEN") or os.environ.get(
        "WIKIPEDIA_ACCESS_TOKEN"
    )
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    for attempt in range(retries):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            if attempt + 1 == retries:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if exc.code == 429:
                try:
                    server_delay = float(retry_after or 0)
                except ValueError:  # Retry-After may be an HTTP date.
                    server_delay = 0.0
                delay = max(server_delay, min(15 * 2**attempt, 120))
            else:
                delay = min(2**attempt, 16)
            time.sleep(delay + random.random() * 0.25)
        except (URLError, TimeoutError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 16) + random.random() * 0.25)
    raise AssertionError("unreachable")


def search_claim(
    proposition: str,
    *,
    page_limit: int,
    include_links: bool = False,
    full_pages: bool = False,
    max_sentences_per_page: int = 24,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "search",
        "gsrsearch": proposition,
        "gsrnamespace": 0,
        "gsrlimit": page_limit,
        "prop": "extracts|info" + ("|links" if include_links else ""),
        "explaintext": 1,
        "inprop": "url",
        "redirects": 1,
        "maxlag": 5,
    }
    if not full_pages:
        params["exintro"] = 1
    if include_links:
        params.update({"plnamespace": 0, "pllimit": 40})
    payload = request_json(params)
    pages = payload.get("query", {}).get("pages", [])
    pages.sort(key=lambda item: int(item.get("index", 10_000)))
    passages: list[dict[str, Any]] = []
    for rank, page in enumerate(pages):
        title = str(page.get("title") or "")
        url = str(page.get("fullurl") or "")
        links = [str(link.get("title")) for link in page.get("links", [])]
        sentences = split_sentences(str(page.get("extract") or ""))
        if full_pages:
            selected = select_document_sentences(
                proposition,
                sentences,
                limit=max_sentences_per_page,
            )
        else:
            selected = [
                (sentence_index, sentence, lexical_relevance(proposition, sentence))
                for sentence_index, sentence in enumerate(sentences)
            ]
        for sentence_index, sentence, lexical_score in selected:
            passages.append({
                "title": title,
                "text": sentence,
                "url": url,
                "page_rank": rank,
                "sentence_index": sentence_index,
                "hop": 0,
                "links": links,
                "lexical_score": lexical_score,
            })
    passages = deduplicate_passages(passages)
    if full_pages:
        passages.sort(
            key=lambda item: (
                float(item.get("lexical_score", 0.0)) - 0.04 * int(item["page_rank"]),
                -int(item["sentence_index"]),
            ),
            reverse=True,
        )
    return passages


def claim_key(record: dict[str, Any]) -> tuple[str, Any, int]:
    return (str(record["dataset"]), record["index"], int(record["claim_index"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="material_assessed")
    parser.add_argument("--dataset-name-contains")
    parser.add_argument("--page-limit", type=int, default=8)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--include-links", action="store_true")
    parser.add_argument(
        "--full-pages",
        action="store_true",
        help="Fetch full article text and retain claim-relevant sentences instead of leads only.",
    )
    parser.add_argument("--max-sentences-per-page", type=int, default=24)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    claims = load_grounded_claims(
        read_jsonl(args.claims),
        variant=args.variant,
        dataset_name_contains=args.dataset_name_contains,
    )
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    claims = [
        claim for position, claim in enumerate(claims)
        if position % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        claims = claims[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, Any, int]] = set()
    if args.output.exists():
        completed = {claim_key(row) for row in read_jsonl(args.output)}

    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode) as handle:
        for position, claim in enumerate(claims, start=1):
            if claim_key(claim) in completed:
                continue
            started = time.perf_counter()
            try:
                passages = search_claim(
                    claim["proposition"],
                    page_limit=args.page_limit,
                    include_links=args.include_links,
                    full_pages=args.full_pages,
                    max_sentences_per_page=args.max_sentences_per_page,
                )
                error = None
            except Exception as exc:  # keep a resumable row-level failure record
                passages = []
                error = f"{type(exc).__name__}: {exc}"
            handle.write(json.dumps({
                **claim,
                "passages": passages,
                "retrieval_seconds": time.perf_counter() - started,
                "error": error,
                "retrieval_config": {
                    "page_limit": args.page_limit,
                    "include_links": args.include_links,
                    "full_pages": args.full_pages,
                    "max_sentences_per_page": args.max_sentences_per_page,
                },
            }, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 50 == 0:
                print(json.dumps({"processed": position, "total": len(claims)}), flush=True)
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
