#!/usr/bin/env python3
"""Build a compact bidirectional relation index from expanded Wikidata cards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable
import zipfile


TARGET_PREDICATES = (
    "applies to jurisdiction",
    "author", "award received", "cast member", "characters", "composer", "conflict",
    "country", "country for sport", "country of origin", "creator", "director",
    "founded by", "head of government", "head of state", "league", "located in",
    "location", "lyrics by", "member of", "member of sports team", "organizer",
    "participant", "participant in", "performer", "place of burial", "position held",
    "screenwriter", "significant event", "sport", "voice actor", "winner", "capital",
)
PREDICATE_IDS = {predicate: number for number, predicate in enumerate(TARGET_PREDICATES, 1)}
YEAR_RE = re.compile(r"^-?([0-9]{4})")


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def year(value: str) -> int | None:
    match = YEAR_RE.match(str(value))
    return int(match.group(1)) if match else None


def collect_labels(cards_paths: list[Path], labels_paths: list[Path]) -> dict[str, str]:
    """Keep labels only for nodes that participate in a selected relation."""
    required: set[str] = set()
    for cards_path in cards_paths:
        for card in jsonl(cards_path):
            selected = [
                claim for claim in card.get("claims", [])
                if claim["property"] in PREDICATE_IDS
            ]
            if not selected:
                continue
            required.add(card["qid"])
            for claim in selected:
                if claim["kind"] == "entity":
                    required.add(claim["value"])
                for qualifier in claim.get("qualifiers", []):
                    if qualifier["kind"] == "entity":
                        required.add(qualifier["value"])
    labels = {}
    for labels_path in labels_paths:
        labels.update({
            row["qid"]: row["label"]
            for row in jsonl(labels_path)
            if row["qid"] in required
        })
    card_labels = {
        card["qid"]: card["label"]
        for cards_path in cards_paths
        for card in jsonl(cards_path)
        if card["qid"] in required
    }
    labels.update({qid: label for qid, label in card_labels.items() if qid in required})
    return labels


def qualifier_fields(
    qualifiers: list[dict[str, Any]], node_ids: dict[str, int]
) -> tuple[int | None, int | None, int | None, int | None, int | None, str]:
    start = end = point = applies_to = for_work = None
    extras = []
    for qualifier in qualifiers:
        predicate = qualifier["property"]
        value = str(qualifier["value"])
        if predicate == "start time":
            start = year(value)
        elif predicate == "end time":
            end = year(value)
        elif predicate == "point in time":
            point = year(value)
        elif predicate == "applies to part" and qualifier["kind"] == "entity":
            applies_to = node_ids.get(value)
        elif predicate == "for work" and qualifier["kind"] == "entity":
            for_work = node_ids.get(value)
        else:
            extras.append(f"{predicate}={value}")
    return start, end, point, applies_to, for_work, "|".join(extras)


def compressed_size(path: Path) -> int:
    archive = path.with_suffix(path.suffix + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        bundle.write(path, path.name)
    size = archive.stat().st_size
    archive.unlink()
    return size


def build(
    cards_path: Path,
    labels_path: Path,
    output: Path,
    *,
    extra_cards: list[Path] | None = None,
    extra_labels: list[Path] | None = None,
) -> dict[str, Any]:
    cards_paths = [cards_path, *(extra_cards or [])]
    labels_paths = [labels_path, *(extra_labels or [])]
    labels = collect_labels(cards_paths, labels_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    connection = sqlite3.connect(output)
    connection.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE node (
            id INTEGER PRIMARY KEY,
            qid TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL
        );
        CREATE TABLE fact (
            subject INTEGER NOT NULL,
            predicate INTEGER NOT NULL,
            object INTEGER,
            literal TEXT,
            start_year INTEGER,
            end_year INTEGER,
            point_year INTEGER,
            applies_to INTEGER,
            for_work INTEGER,
            extras TEXT NOT NULL
        );
    """)
    ordered_labels = sorted(labels.items())
    connection.executemany(
        "INSERT INTO node(id,qid,label) VALUES (?,?,?)",
        ((number, qid, label) for number, (qid, label) in enumerate(ordered_labels, 1)),
    )
    node_ids = {qid: number for number, (qid, _) in enumerate(ordered_labels, 1)}
    facts = []
    counts: Counter[str] = Counter()
    skipped_missing_label = 0
    seen_facts: set[tuple[Any, ...]] = set()
    for cards_file in cards_paths:
        for card in jsonl(cards_file):
            subject = node_ids.get(card["qid"])
            if subject is None:
                continue
            for claim in card.get("claims", []):
                predicate = claim["property"]
                if predicate not in PREDICATE_IDS:
                    continue
                object_id = None
                literal = None
                if claim["kind"] == "entity":
                    object_id = node_ids.get(claim["value"])
                    if object_id is None:
                        skipped_missing_label += 1
                        continue
                else:
                    literal = str(claim["value"])
                fields = qualifier_fields(claim.get("qualifiers", []), node_ids)
                fact = (subject, PREDICATE_IDS[predicate], object_id, literal, *fields)
                if fact in seen_facts:
                    continue
                seen_facts.add(fact)
                facts.append(fact)
                counts[predicate] += 1
                if len(facts) >= 20_000:
                    connection.executemany(
                        "INSERT INTO fact VALUES (?,?,?,?,?,?,?,?,?,?)", facts
                    )
                    facts.clear()
    if facts:
        connection.executemany("INSERT INTO fact VALUES (?,?,?,?,?,?,?,?,?,?)", facts)
    connection.executescript("""
        CREATE INDEX fact_subject_predicate ON fact(subject, predicate);
        CREATE INDEX fact_object_predicate ON fact(object, predicate) WHERE object IS NOT NULL;
        VACUUM;
    """)
    fact_rows = connection.execute("SELECT COUNT(*) FROM fact").fetchone()[0]
    qualifier_counts = {
        "start_year": connection.execute(
            "SELECT COUNT(*) FROM fact WHERE start_year IS NOT NULL"
        ).fetchone()[0],
        "end_year": connection.execute(
            "SELECT COUNT(*) FROM fact WHERE end_year IS NOT NULL"
        ).fetchone()[0],
        "point_year": connection.execute(
            "SELECT COUNT(*) FROM fact WHERE point_year IS NOT NULL"
        ).fetchone()[0],
        "for_work": connection.execute(
            "SELECT COUNT(*) FROM fact WHERE for_work IS NOT NULL"
        ).fetchone()[0],
    }
    connection.close()
    report = {
        "nodes": len(node_ids),
        "facts": fact_rows,
        "predicates": dict(counts),
        "qualifiers": qualifier_counts,
        "skipped_missing_label": skipped_missing_label,
        "sqlite_bytes": output.stat().st_size,
        "zip_bytes": compressed_size(output),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--extra-cards", type=Path, action="append", default=[])
    parser.add_argument("--extra-labels", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        args.cards, args.labels, args.output,
        extra_cards=args.extra_cards, extra_labels=args.extra_labels,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
