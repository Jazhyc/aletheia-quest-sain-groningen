#!/usr/bin/env python3
"""Search Wikipedia titles from label-blind abstention queries."""

from __future__ import annotations

import argparse
import fcntl
import html
import json
from pathlib import Path
import random
import re
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
ENTITY_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][\w'’.-]*)(?:\s+(?:(?:of|the|and|de|van|von)\s+)?"
    r"[A-Z][\w'’.-]*)+"
)
QUOTED_PHRASE_RE = re.compile(r'''["“]([^"”]{3,80})["”]''')
QUERY_STOPWORDS = {
    "about", "after", "author", "before", "behind", "biography", "birth",
    "born", "date", "does", "from", "give", "into", "name", "real",
    "such", "that", "their", "these", "this", "what", "when", "where",
    "which", "with", "would",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_abstentions(
    rows: list[dict[str, Any]],
    *,
    candidate_state: str,
    sample: int | None,
    seed: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Choose a deterministic, label-blind retrieval subset."""
    selected = [row for row in rows if row.get("decision") == "ABSTAIN"]
    if candidate_state == "empty":
        selected = [row for row in selected if not row.get("candidates")]
    elif candidate_state == "nonempty":
        selected = [row for row in selected if row.get("candidates")]
    if sample is not None:
        rng = random.Random(seed)
        selected = rng.sample(selected, k=min(sample, len(selected)))
        selected.sort(key=claim_key)
    if limit is not None:
        selected = selected[:limit]
    return selected


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


def broaden_search_query(*texts: str, max_terms: int = 6) -> str:
    """Build a label-blind OR query from named phrases and distinctive tokens."""
    phrases = []
    for text in texts:
        phrases.extend(QUOTED_PHRASE_RE.findall(text))
        phrases.extend(ENTITY_PHRASE_RE.findall(text))
    normalized = []
    seen = set()
    for phrase in phrases:
        value = " ".join(phrase.strip(" ?.,:;!()[]{}").split())
        key = value.casefold()
        if len(value.split()) < 2 or key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    # Longer names and titles are generally more discriminative than short
    # sentence-initial capitalized fragments.
    normalized.sort(key=lambda value: (len(value.split()), len(value)), reverse=True)
    terms = [f'"{value}"' for value in normalized[:max_terms]]
    if len(terms) < 2:
        tokens = []
        for text in texts:
            for token in re.findall(r"\b[\w'’.-]{4,}\b", text):
                key = token.casefold().strip(".-")
                if key in QUERY_STOPWORDS or key in seen:
                    continue
                seen.add(key)
                tokens.append(token)
        terms.extend(tokens[: max_terms - len(terms)])
    return " OR ".join(terms[:max_terms])


def search_lead_pages(
    query: str,
    proposition: str,
    question: str,
    *,
    page_limit: int,
    max_sentences: int,
) -> list[dict[str, Any]]:
    """Retrieve several page leads from one broadened MediaWiki request."""
    broad_query = broaden_search_query(question, query, proposition)
    payload = request_json({
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "search",
        "gsrsearch": broad_query or query,
        "gsrnamespace": 0,
        "gsrlimit": page_limit,
        "prop": "extracts|info",
        "explaintext": 1,
        "exintro": 1,
        "exlimit": page_limit,
        "inprop": "url",
        "redirects": 1,
        "maxlag": 5,
    })
    passages = []
    pages = sorted(
        payload.get("query", {}).get("pages", []),
        key=lambda page: int(page.get("index", page_limit + 1)),
    )
    per_page = max(2, max_sentences // max(1, min(page_limit, len(pages))))
    for fallback_rank, page in enumerate(pages):
        page_rank = int(page.get("index", fallback_rank + 1)) - 1
        sentences = split_sentences(str(page.get("extract") or ""))
        selected = select_document_sentences(
            proposition,
            sentences,
            limit=per_page,
        )
        lead = [
            (index, sentence, 0.0)
            for index, sentence in enumerate(sentences[:2])
        ]
        selected_by_index = {
            index: (index, sentence, score)
            for index, sentence, score in [*lead, *selected]
        }
        for sentence_index, sentence, lexical_score in selected_by_index.values():
            passages.append({
                "title": str(page.get("title") or ""),
                "text": sentence,
                "url": str(page.get("fullurl") or ""),
                "page_rank": page_rank,
                "sentence_index": sentence_index,
                "hop": 1,
                "links": [],
                "lexical_score": lexical_score,
                "query_source": "gpt_oss_entity_or_multi_page_leads",
                "search_query": broad_query,
            })
    passages.sort(
        key=lambda item: (
            float(item.get("lexical_score", 0.0))
            - 0.02 * int(item.get("page_rank", 0)),
            -int(item.get("sentence_index", 0)),
        ),
        reverse=True,
    )
    return passages[:max_sentences]


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
    parser.add_argument(
        "--multi-page-leads",
        action="store_true",
        help="Fetch several page introductions using a broadened entity-phrase query.",
    )
    parser.add_argument("--max-sentences", type=int, default=24)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--candidate-state",
        choices=("any", "empty", "nonempty"),
        default="any",
        help="Restrict retrieval to abstentions by whether their initial audit had candidates.",
    )
    parser.add_argument(
        "--sample-abstentions",
        type=int,
        help="Deterministically sample this many abstaining claims before retrieval.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.full_pages and args.multi_page_leads:
        parser.error("--full-pages and --multi-page-leads are mutually exclusive")

    rows = select_abstentions(
        read_jsonl(args.audit),
        candidate_state=args.candidate_state,
        sample=args.sample_abstentions,
        seed=args.seed,
        limit=args.limit,
    )
    existing = read_jsonl(args.output) if args.output.exists() else []
    completed = {claim_key(row) for row in existing if not row.get("error")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"another retrieval process is writing {args.output}") from exc
        for position, row in enumerate(rows, start=1):
            if claim_key(row) in completed:
                continue
            started = time.perf_counter()
            passages = []
            error = None
            try:
                query = str(row.get("query") or row["proposition"])
                if args.multi_page_leads:
                    passages = search_lead_pages(
                        query,
                        str(row["proposition"]),
                        str(row.get("question") or ""),
                        page_limit=args.page_limit,
                        max_sentences=args.max_sentences,
                    )
                elif args.full_pages:
                    passages = search_full_page(
                        query,
                        str(row["proposition"]),
                        max_sentences=args.max_sentences,
                    )
                else:
                    passages = search_titles(query, args.page_limit)
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
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
