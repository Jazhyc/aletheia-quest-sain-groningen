#!/usr/bin/env python3
"""Fetch full Wikipedia pages for titles already found by question retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.retrieve_wikipedia import request_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_requested_pages(
    requested: list[str], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Map normalized and redirected API results back to every requested title."""
    query = payload.get("query", {})
    aliases = {
        str(item["from"]): str(item["to"])
        for field in ("normalized", "redirects")
        for item in query.get(field, [])
    }
    pages = {
        str(page.get("title")): page
        for page in query.get("pages", [])
    }
    output = []
    for title in requested:
        canonical = title
        seen = set()
        while canonical in aliases and canonical not in seen:
            seen.add(canonical)
            canonical = aliases[canonical]
        page = pages.get(canonical, {})
        output.append({
            "requested_title": title,
            "canonical_title": str(page.get("title") or canonical),
            "url": str(page.get("fullurl") or ""),
            "extract": str(page.get("extract") or ""),
            "missing": bool(page.get("missing")) or not page,
            "error": None if page.get("extract") or page.get("missing") else "empty extract",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Must remain 1: MediaWiki limits whole-article extracts to one page per call.",
    )
    parser.add_argument(
        "--max-page-rank",
        type=int,
        help="Only fetch titles up to this zero-based search rank (0 is the cheap first pass).",
    )
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    retrieval = read_jsonl(args.retrieval)
    if args.batch_size != 1:
        raise ValueError("whole-article MediaWiki extracts require --batch-size 1")
    titles = sorted({
        str(passage.get("title") or "").strip()
        for row in retrieval
        for passage in row.get("passages") or []
        if str(passage.get("title") or "").strip()
        and (
            args.max_page_rank is None
            or int(passage.get("page_rank", 0)) <= args.max_page_rank
        )
    })
    if args.limit is not None:
        titles = titles[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(args.output) if args.output.exists() else []
    successful = {
        str(row["requested_title"]): row
        for row in existing
        if not row.get("error")
    }
    pending = [title for title in titles if title not in successful]
    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode) as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            try:
                payload = request_json({
                    "action": "query",
                    "format": "json",
                    "formatversion": 2,
                    "titles": "|".join(batch),
                    "prop": "extracts|info",
                    "explaintext": 1,
                    "inprop": "url",
                    "redirects": 1,
                    "maxlag": 5,
                })
                rows = resolve_requested_pages(batch, payload)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                rows = [
                    {
                        "requested_title": title,
                        "canonical_title": title,
                        "url": "",
                        "extract": "",
                        "missing": False,
                        "error": message,
                    }
                    for title in batch
                ]
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({
                "processed": min(start + len(batch), len(pending)),
                "pending": len(pending),
                "total_titles": len(titles),
            }), flush=True)
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
