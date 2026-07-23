#!/usr/bin/env python3
"""Cache short Wikipedia search passages for a labeled public split."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import merge_messages
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import load_split_config, load_labels


SEARCH_URL = "https://en.wikipedia.org/w/api.php"
TAG_RE = re.compile(r"<[^>]+>")


def final_user_query(messages: Any, max_chars: int = 500) -> str:
    merged = merge_messages(messages)
    query = next(
        (message["content"] for message in reversed(merged) if message["role"] == "user"),
        "",
    )
    query = " ".join(query.split())
    return query[-max_chars:]


def clean_excerpt(text: Any) -> str:
    return " ".join(html.unescape(TAG_RE.sub("", str(text or ""))).split())


def search_wikipedia(session: Any, query: str, limit: int) -> list[dict[str, str]]:
    if not query:
        return []
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "formatversion": 2,
    }
    for attempt in range(6):
        response = session.get(SEARCH_URL, params=params, timeout=30)
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else min(60.0, 5.0 * 2**attempt)
        print(f"Wikipedia rate limit; retrying in {delay:.1f}s", flush=True)
        time.sleep(delay)
    response.raise_for_status()
    passages = []
    for page in response.json().get("query", {}).get("search", []):
        text = clean_excerpt(page.get("snippet"))
        if text:
            passages.append({"title": str(page.get("title", "")), "text": text})
    return passages


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def reusable_queries(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """Index successful cached searches so repeated questions need one API call."""
    return {
        str(record.get("query", "")): list(record.get("passages") or [])
        for record in records
        if record.get("query") and record.get("error") is None
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="validation", choices=["train", "validation", "test"]
    )
    parser.add_argument("--scenario", default="varied-deception")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-results", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=0.05)
    args = parser.parse_args()

    from datasets import load_dataset
    import requests

    existing: dict[tuple[str, Any], dict[str, Any]] = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                existing[(record["dataset"], record["index"])] = record
    query_cache = reusable_queries(list(existing.values()))

    session = requests.Session()
    session.headers["User-Agent"] = "AletheiasQuestResearch/1.0 (deception detection research)"
    access_token = os.environ.get("WIKIMEDIA_ACCESS_TOKEN")
    if access_token:
        session.headers["Authorization"] = f"Bearer {access_token}"
        print("using authenticated Wikimedia access", flush=True)
    records: list[dict[str, Any]] = []
    for cfg in load_split_config(args.splits_dir / f"dry.{args.split}.yaml", ROOT):
        if args.scenario not in cfg.name:
            continue
        labels = load_labels(cfg)
        wanted = set(labels["index"].tolist())
        dataset = load_dataset(cfg.name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        for row in dataset:
            if row["index"] not in wanted:
                continue
            key = (cfg.name, row["index"])
            query = final_user_query(row["messages"])
            cached = existing.get(key)
            if (
                cached is not None
                and cached.get("query") == query
                and cached.get("error") is None
            ):
                record = cached
            elif query in query_cache:
                record = {
                    "dataset": cfg.name,
                    "index": row["index"],
                    "query": query,
                    "passages": query_cache[query],
                    "error": None,
                }
            else:
                try:
                    passages = search_wikipedia(session, query, args.limit_results)
                    error = None
                except Exception as exc:
                    passages = []
                    error = f"{type(exc).__name__}: {exc}"
                record = {
                    "dataset": cfg.name,
                    "index": row["index"],
                    "query": query,
                    "passages": passages,
                    "error": error,
                }
                if error is None:
                    query_cache[query] = passages
                time.sleep(args.delay_seconds)
            records.append(record)
            if len(records) % 25 == 0:
                write_records(args.output, records)
                print(f"cached {len(records)} rows", flush=True)

    write_records(args.output, records)
    print(
        f"wrote {len(records)} rows to {args.output}; "
        f"with_passages={sum(bool(record['passages']) for record in records)} "
        f"errors={sum(record['error'] is not None for record in records)}"
    )


if __name__ == "__main__":
    main()
