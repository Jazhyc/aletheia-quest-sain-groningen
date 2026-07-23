#!/usr/bin/env python3
"""Refetch linked entities with uncapped relation histories and useful qualifiers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import requests

from experiments.wikidata_rag.build_relation_index import TARGET_PREDICATES
from experiments.wikidata_rag.evaluate_retrieval import (
    PROPERTY_LABELS,
    QUALIFIER_LABELS,
    datavalue,
    fetch_entities,
)


TARGET_PIDS = {
    pid for pid, label in PROPERTY_LABELS.items() if label in TARGET_PREDICATES
}
HISTORY_PIDS = {"P39", "P54", "P161", "P166", "P463", "P725", "P1346"}


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def relation_card(qid: str, entity: dict[str, Any]) -> dict[str, Any]:
    claims = []
    for pid, statements in entity.get("claims", {}).items():
        if pid not in TARGET_PIDS:
            continue
        cap = 96 if pid in HISTORY_PIDS else 16
        retained = 0
        for statement in statements:
            if retained >= cap or statement.get("rank") == "deprecated":
                continue
            value = datavalue(statement.get("mainsnak", {}))
            if not value:
                continue
            kind, parsed = value
            qualifiers = []
            for qualifier_pid, snaks in statement.get("qualifiers", {}).items():
                if qualifier_pid not in QUALIFIER_LABELS:
                    continue
                for snak in snaks[:4]:
                    qualifier = datavalue(snak)
                    if qualifier:
                        qualifier_kind, qualifier_value = qualifier
                        qualifiers.append({
                            "property": QUALIFIER_LABELS[qualifier_pid],
                            "kind": qualifier_kind,
                            "value": qualifier_value,
                        })
            claims.append({
                "property": PROPERTY_LABELS[pid], "kind": kind, "value": parsed,
                "qualifiers": qualifiers,
            })
            retained += 1
    return {
        "qid": qid,
        "label": entity.get("labels", {}).get("en", {}).get("value", qid),
        "claims": claims,
    }


def fetch_cards(
    session: requests.Session,
    qids: Iterable[str],
    path: Path,
    *,
    delay: float,
) -> dict[str, dict[str, Any]]:
    existing = {row["qid"]: row for row in load_jsonl(path)}
    missing = [qid for qid in dict.fromkeys(qids) if qid not in existing]
    batches = list(chunks(missing, 50))
    for number, batch in enumerate(batches, 1):
        entities = fetch_entities(session, batch, delay)
        cards = [relation_card(qid, entities[qid]) for qid in batch if qid in entities]
        append_jsonl(path, cards)
        existing.update({card["qid"]: card for card in cards})
        if number % 10 == 0 or number == len(batches):
            print(f"card batches {number}/{len(batches)} cards={len(existing)}", flush=True)
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delay-seconds", type=float, default=0.1)
    args = parser.parse_args()
    seeds = load_jsonl(args.seed_cache)
    qids = sorted({
        qid for row in seeds for field in ("qids", "answer_qids") for qid in row.get(field, [])
    })
    session = requests.Session()
    session.headers["User-Agent"] = "AletheiaRelationEnrichment/0.1 (research)"
    if os.environ.get("WIKIMEDIA_ACCESS_TOKEN"):
        session.headers["Authorization"] = f"Bearer {os.environ['WIKIMEDIA_ACCESS_TOKEN']}"
    cards_path = args.output_dir / "cards.jsonl"
    cards = fetch_cards(session, qids, cards_path, delay=args.delay_seconds)

    # Fetch position/office nodes as a constrained second hop. Their jurisdiction
    # claims support office/monarch routing; their labels also ground qualifiers.
    office_qids = sorted({
        claim["value"] for card in cards.values() for claim in card["claims"]
        if claim["property"] == "position held" and claim["kind"] == "entity"
    })
    cards = fetch_cards(session, office_qids, cards_path, delay=args.delay_seconds)
    referenced = sorted({
        item["value"]
        for card in cards.values()
        for claim in card["claims"]
        for item in [claim, *claim.get("qualifiers", [])]
        if item["kind"] == "entity"
    })
    labels_path = args.output_dir / "labels.jsonl"
    known = {row["qid"] for row in load_jsonl(labels_path)} | set(cards)
    missing = [qid for qid in referenced if qid not in known]
    batches = list(chunks(missing, 50))
    for number, batch in enumerate(batches, 1):
        entities = fetch_entities(session, batch, args.delay_seconds)
        rows = [{
            "qid": qid,
            "label": entity.get("labels", {}).get("en", {}).get("value", qid),
        } for qid, entity in entities.items()]
        append_jsonl(labels_path, rows)
        if number % 10 == 0 or number == len(batches):
            print(f"label batches {number}/{len(batches)}", flush=True)
    session.close()
    report = {
        "seed_qids": len(qids), "office_qids": len(office_qids),
        "cards": len(cards), "referenced_qids": len(referenced),
        "target_properties": len(TARGET_PIDS),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
