#!/usr/bin/env python3
"""Refetch a fixed QID universe with the expanded fact schema and build SQLite.

The input card cache supplies only entity selection, source, and popularity.  Its
old claims are deliberately ignored so this remains a clean, resumable schema
upgrade rather than a second pageview crawl.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.wikidata_rag.build_broad_index import (
    append_cards,
    build_sqlite,
    chunks,
    compressed_size,
    fetch_card_batches,
    load_cards,
    referenced_qids,
    request_json,
    save_json,
)
from experiments.wikidata_rag.evaluate_retrieval import (
    ENTITY_URL,
    PROPERTY_LABELS,
    QUALIFIER_LABELS,
)


def load_seed_cards(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                labels[row["qid"]] = row["label"]
    return labels


def fetch_label_batches(
    session: requests.Session,
    qids: Iterable[str],
    path: Path,
    labels: dict[str, str],
    *,
    batch_size: int,
    delay: float,
    workers: int,
) -> None:
    missing = [qid for qid in dict.fromkeys(qids) if qid not in labels]
    batches = list(chunks(missing, batch_size))

    def fetch_one(batch: list[str]) -> tuple[list[str], dict[str, Any]]:
        worker_session = requests.Session()
        worker_session.headers.update(session.headers)
        try:
            data = request_json(worker_session, ENTITY_URL, {
                "action": "wbgetentities", "ids": "|".join(batch), "languages": "en",
                "languagefallback": 1, "props": "labels", "format": "json",
                "formatversion": 2,
            }, delay)
            return batch, data.get("entities", {})
        finally:
            worker_session.close()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for number, (batch, entities) in enumerate(executor.map(fetch_one, batches), 1):
            rows = []
            for qid in batch:
                entity = entities.get(qid, {})
                label = entity.get("labels", {}).get("en", {}).get("value", "")
                if label:
                    labels[qid] = label
                    rows.append({"qid": qid, "label": label})
            append_cards(path, rows)
            if number % 20 == 0 or number == len(batches):
                print(f"label batches {number}/{len(batches)} labels={len(labels)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-cards", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--delay-seconds", type=float, default=0.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed = load_seed_cards(args.seed_cards)
    scores = Counter({row["qid"]: int(row.get("score", 0)) for row in seed})
    source_by_qid = {row["qid"]: row.get("source", "seed") for row in seed}

    session = requests.Session()
    session.headers["User-Agent"] = "AletheiaWikidataExpandedIndex/0.2 (deception detection research)"
    if os.environ.get("WIKIMEDIA_ACCESS_TOKEN"):
        session.headers["Authorization"] = f"Bearer {os.environ['WIKIMEDIA_ACCESS_TOKEN']}"

    cards_path = cache_dir / "cards.jsonl"
    cards = load_cards(cards_path)
    for source in dict.fromkeys(source_by_qid.values()):
        qids = [qid for qid, value in source_by_qid.items() if value == source]
        fetch_card_batches(
            session, qids, cards_path, cards, source, scores,
            args.batch_size, args.delay_seconds, args.workers,
        )

    card_labels = {qid: card["label"] for qid, card in cards.items()}
    referenced = referenced_qids(cards.values())
    labels_path = cache_dir / "object_labels.jsonl"
    labels = load_labels(labels_path)
    fetch_label_batches(
        session,
        (qid for qid, _ in referenced.most_common() if qid not in card_labels),
        labels_path,
        labels,
        batch_size=args.batch_size,
        delay=args.delay_seconds,
        workers=args.workers,
    )
    session.close()

    database = args.output_dir / "wikidata.sqlite"
    build_sqlite(database, cards, labels)
    rendered_facts = sum(
        min(32, len(card.get("claims", []))) for card in cards.values()
    )
    report = {
        "seed_cards": len(seed),
        "cards": len(cards),
        "properties": len(PROPERTY_LABELS),
        "qualifier_properties": len(QUALIFIER_LABELS),
        "raw_claims": sum(len(card.get("claims", [])) for card in cards.values()),
        "rendered_fact_upper_bound": rendered_facts,
        "referenced_qids": len(referenced),
        "extra_object_labels": len(labels),
        "sqlite_bytes": database.stat().st_size,
        "zip_bytes": compressed_size(database),
    }
    save_json(args.output_dir / "build_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
