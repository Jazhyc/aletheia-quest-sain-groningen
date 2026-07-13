#!/usr/bin/env python3
"""Build a broad, competition-independent CC0 Wikidata SQLite index."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Iterable
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.wikidata_rag.evaluate_retrieval import (
    entity_stub,
    fetch_entities,
)


PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
    "en.wikipedia.org/all-access/{year:04d}/{month:02d}/all-days"
)
PAGEVIEWS_DAILY_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
    "en.wikipedia.org/all-access/{year:04d}/{month:02d}/{day:02d}"
)
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
META_TYPES = {
    "Q4167410",   # Wikimedia disambiguation page
    "Q13406463",  # Wikimedia list article
    "Q11266439",  # Wikimedia template
    "Q4663903",   # Wikimedia portal
    "Q15184295",  # Wikimedia module
    "Q14204246",  # Wikimedia project page
    "Q17362920",  # duplicated page
}


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    if (sy, sm) > (ey, em):
        raise ValueError("start month must not follow end month")
    months = []
    year, month = sy, sm
    while (year, month) <= (ey, em):
        months.append((year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


def daily_snapshots(start: str, end: str, day_step: int) -> list[tuple[int, int, int]]:
    if day_step < 1:
        raise ValueError("day_step must be positive")
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    current = date(sy, sm, 1)
    finish = date(ey + (em == 12), 1 if em == 12 else em + 1, 1) - timedelta(days=1)
    snapshots = []
    while current <= finish:
        snapshots.append((current.year, current.month, current.day))
        current += timedelta(days=day_step)
    return snapshots


def request_json(
    session: Any,
    url: str,
    params: dict[str, Any] | None,
    delay: float,
    *,
    allow_not_found: bool = False,
) -> dict[str, Any]:
    for attempt in range(10):
        try:
            response = session.get(url, params=params, timeout=60)
        except requests.RequestException:
            if attempt == 9:
                raise
            time.sleep(min(60.0, 2.0**attempt))
            continue
        if response.status_code == 404 and allow_not_found:
            return {}
        if response.status_code not in {429, 502, 503, 504}:
            response.raise_for_status()
            data = response.json()
            error = data.get("error", {})
            if error.get("code") in {"maxlag", "ratelimited"}:
                retry = response.headers.get("Retry-After")
                time.sleep(float(retry) if retry else min(60.0, 2.0**attempt))
                continue
            if error:
                raise RuntimeError(f"API error: {error}")
            if delay:
                time.sleep(delay)
            return data
        retry = response.headers.get("Retry-After")
        time.sleep(float(retry) if retry else min(60.0, 2.0**attempt))
    response.raise_for_status()
    raise RuntimeError("Wikimedia API remained throttled after 10 retries")


def load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True))
    temporary.replace(path)


def collect_pageviews(
    session: Any,
    snapshots: list[tuple[int, int] | tuple[int, int, int]],
    cache_dir: Path,
    delay: float,
) -> Counter[str]:
    title_views: Counter[str] = Counter()
    for number, snapshot in enumerate(snapshots, 1):
        year, month = snapshot[:2]
        day = snapshot[2] if len(snapshot) == 3 else None
        suffix = f"{year:04d}-{month:02d}" + (f"-{day:02d}" if day is not None else "")
        path = cache_dir / "pageviews" / f"{suffix}.json"
        if path.exists():
            articles = json.loads(path.read_text())
        else:
            url = (
                PAGEVIEWS_DAILY_URL.format(year=year, month=month, day=day)
                if day is not None
                else PAGEVIEWS_URL.format(year=year, month=month)
            )
            data = request_json(
                session,
                url,
                None,
                delay,
                allow_not_found=True,
            )
            items = data.get("items", [])
            articles = items[0].get("articles", []) if items else []
            save_json(path, articles)
        for article in articles:
            title = str(article.get("article", "")).replace("_", " ").strip()
            if title and not title.startswith(("Special:", "Main Page")):
                title_views[title] += int(article.get("views", 0))
        if number % 12 == 0 or number == len(snapshots):
            print(f"pageviews {number}/{len(snapshots)} unique_titles={len(title_views)}", flush=True)
    return title_views


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def page_qids(data: dict[str, Any]) -> dict[str, str]:
    """Map requested and canonical titles to QIDs from an Action API response."""
    transforms = {}
    query = data.get("query", {})
    for kind in ("normalized", "redirects"):
        for row in query.get(kind, []):
            transforms[row["from"]] = row["to"]
    result = {}
    for page in query.get("pages", []):
        qid = page.get("pageprops", {}).get("wikibase_item")
        if qid:
            result[page["title"]] = qid
    for source in list(transforms):
        current = source
        seen = set()
        while current in transforms and current not in seen:
            seen.add(current)
            current = transforms[current]
        if current in result:
            result[source] = result[current]
    return result


def map_titles_to_qids(
    session: Any,
    title_views: Counter[str],
    cache_dir: Path,
    batch_size: int,
    delay: float,
) -> Counter[str]:
    titles = [title for title, _ in title_views.most_common()]
    qid_scores: Counter[str] = Counter(load_json(cache_dir / "qid_scores.json", {}))
    completed = int(load_json(cache_dir / "title_progress.json", 0))
    batches = list(chunks(titles, batch_size))
    for batch_number, batch in enumerate(batches):
        if batch_number < completed:
            continue
        data = request_json(session, WIKIPEDIA_API, {
            "action": "query", "titles": "|".join(batch), "redirects": 1,
            "prop": "pageprops", "ppprop": "wikibase_item", "format": "json",
            "formatversion": 2,
        }, delay)
        mapped = page_qids(data)
        for title in batch:
            qid = mapped.get(title)
            if qid:
                qid_scores[qid] += title_views[title]
        completed = batch_number + 1
        if completed % 20 == 0 or completed == len(batches):
            save_json(cache_dir / "qid_scores.json", dict(qid_scores))
            save_json(cache_dir / "title_progress.json", completed)
            print(f"title batches {completed}/{len(batches)} qids={len(qid_scores)}", flush=True)
    return qid_scores


def load_cards(path: Path) -> dict[str, dict[str, Any]]:
    cards = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                card = json.loads(line)
                cards[card["qid"]] = card
    return cards


def append_cards(path: Path, cards: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for card in cards:
            handle.write(json.dumps(card) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def is_meta(stub: dict[str, Any]) -> bool:
    return any(
        claim["property"] == "instance of" and claim["value"] in META_TYPES
        for claim in stub.get("claims", [])
    )


def fetch_card_batches(
    session: Any,
    qids: list[str],
    cards_path: Path,
    cards: dict[str, dict[str, Any]],
    source: str,
    scores: Counter[str],
    batch_size: int,
    delay: float,
    workers: int,
) -> None:
    missing = [qid for qid in qids if qid not in cards]
    batches = list(chunks(missing, batch_size))
    total = len(batches)

    def fetch_one(batch: list[str]) -> tuple[list[str], dict[str, dict[str, Any]]]:
        worker_session = type(session)()
        worker_session.headers.update(session.headers)
        try:
            return batch, fetch_entities(worker_session, batch, delay)
        finally:
            worker_session.close()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for number, (batch, entities) in enumerate(executor.map(fetch_one, batches), 1):
            new_cards = []
            for qid in batch:
                entity = entities.get(qid)
                if not entity or entity.get("missing") is not None:
                    continue
                stub = entity_stub(entity)
                if not stub["label"] or is_meta(stub):
                    continue
                card = {"qid": qid, "source": source, "score": int(scores.get(qid, 0)), **stub}
                cards[qid] = card
                new_cards.append(card)
            append_cards(cards_path, new_cards)
            if number % 20 == 0 or number == total:
                print(f"{source} entity batches {number}/{total} cards={len(cards)}", flush=True)


def referenced_qids(cards: Iterable[dict[str, Any]]) -> Counter[str]:
    refs: Counter[str] = Counter()
    for card in cards:
        for claim in card.get("claims", []):
            if claim["kind"] == "entity":
                refs[claim["value"]] += 1
            for qualifier in claim.get("qualifiers", []):
                if qualifier["kind"] == "entity":
                    refs[qualifier["value"]] += 1
    return refs


def format_facts(card: dict[str, Any], labels: dict[str, str]) -> str:
    facts = []
    for claim in card.get("claims", []):
        value = claim["value"]
        if claim["kind"] == "entity":
            value = labels.get(value, "")
            if not value:
                continue
        qualifiers = []
        for qualifier in claim.get("qualifiers", []):
            qualifier_value = qualifier["value"]
            if qualifier["kind"] == "entity":
                qualifier_value = labels.get(qualifier_value, "")
                if not qualifier_value:
                    continue
            qualifiers.append(f"{qualifier['property']}: {qualifier_value}")
        suffix = f" [{', '.join(qualifiers)}]" if qualifiers else ""
        fact = f"{claim['property']}: {value}{suffix}"
        if fact not in facts:
            facts.append(fact)
        if len(facts) == 32:
            break
    return "; ".join(facts)


def build_sqlite(
    path: Path,
    cards: dict[str, dict[str, Any]],
    extra_labels: dict[str, str] | None = None,
) -> None:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = {qid: card["label"] for qid, card in cards.items()}
    labels.update(extra_labels or {})
    connection = sqlite3.connect(path)
    connection.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE entity (
            qid TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL,
            aliases TEXT NOT NULL,
            description TEXT NOT NULL,
            facts TEXT NOT NULL,
            source TEXT NOT NULL,
            popularity INTEGER NOT NULL
        );
    """)
    rows = []
    for qid, card in cards.items():
        rows.append((
            qid, card["label"], "; ".join(card.get("aliases", [])),
            card.get("description", ""), format_facts(card, labels),
            card["source"], int(card.get("score", 0)),
        ))
    connection.executemany(
        "INSERT INTO entity(qid,label,aliases,description,facts,source,popularity) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    connection.executescript("""
        CREATE VIRTUAL TABLE entity_fts USING fts5(
            label, aliases, description, facts,
            content='entity', content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 2'
        );
        INSERT INTO entity_fts(entity_fts) VALUES('rebuild');
        CREATE INDEX entity_popularity ON entity(popularity DESC);
        VACUUM;
    """)
    connection.close()


def compressed_size(path: Path) -> int:
    archive = path.with_suffix(path.suffix + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(path, path.name)
    size = archive.stat().st_size
    archive.unlink()
    return size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2021-01")
    parser.add_argument("--end", default=f"{date.today().year}-{date.today().month - 1:02d}")
    parser.add_argument("--max-cards", type=int, default=30_000)
    parser.add_argument("--day-step", type=int, default=0, help="Use daily snapshots at this interval; 0 uses monthly snapshots")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--delay-seconds", type=float, default=0.05)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    session = requests.Session()
    session.headers["User-Agent"] = "AletheiaWikidataIndex/0.1 (deception detection research)"
    if os.environ.get("WIKIMEDIA_ACCESS_TOKEN"):
        session.headers["Authorization"] = f"Bearer {os.environ['WIKIMEDIA_ACCESS_TOKEN']}"

    snapshots = (
        daily_snapshots(args.start, args.end, args.day_step)
        if args.day_step else month_range(args.start, args.end)
    )
    title_views = collect_pageviews(session, snapshots, cache_dir, args.delay_seconds)
    title_views = Counter(dict(title_views.most_common(args.max_cards * 3)))
    qid_scores = map_titles_to_qids(session, title_views, cache_dir, args.batch_size, args.delay_seconds)
    cards_path = cache_dir / "cards.jsonl"
    cards = load_cards(cards_path)
    seed_qids = [qid for qid, _ in qid_scores.most_common(args.max_cards)]
    fetch_card_batches(
        session, seed_qids, cards_path, cards, "pageview", qid_scores,
        args.batch_size, args.delay_seconds, args.workers,
    )
    if len(cards) < args.max_cards:
        seed_set = set(seed_qids)
        refs = referenced_qids(card for qid, card in cards.items() if qid in seed_set)
        related = [qid for qid, _ in refs.most_common() if qid not in cards]
        fetch_card_batches(
            session, related[:args.max_cards - len(cards)], cards_path, cards,
            "one_hop", refs, args.batch_size, args.delay_seconds, args.workers,
        )

    selected = dict(list(cards.items())[:args.max_cards])
    database = args.output_dir / "wikidata.sqlite"
    build_sqlite(database, selected)
    report = {
        "snapshots": len(snapshots),
        "day_step": args.day_step,
        "unique_titles": len(title_views),
        "seed_qids": len(qid_scores),
        "cards": len(selected),
        "pageview_cards": sum(card["source"] == "pageview" for card in selected.values()),
        "one_hop_cards": sum(card["source"] == "one_hop" for card in selected.values()),
        "sqlite_bytes": database.stat().st_size,
        "zip_bytes": compressed_size(database),
    }
    save_json(args.output_dir / "build_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
