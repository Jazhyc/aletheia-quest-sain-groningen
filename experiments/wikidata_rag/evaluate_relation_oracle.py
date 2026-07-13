#!/usr/bin/env python3
"""Measure fact-retrieval ceilings before training a compact fallback ranker."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Iterable

from experiments.wikidata_rag.build_relation_index import TARGET_PREDICATES
from experiments.wikidata_rag.claim_retrieval import (
    extract_claim_query,
    query_candidate_entities,
)
from experiments.wikidata_rag.evaluate_retrieval import score_record
from experiments.wikidata_rag.relation_retrieval import (
    node_ids,
    query_facts,
    render_fact,
)


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def broad_qids(connection: sqlite3.Connection, conversation: str) -> list[str]:
    """Return every exact/alias entity linked from a bounded claim span."""
    claim = extract_claim_query(conversation)
    selected: dict[str, None] = {}
    for span in (*claim.subjects, *claim.answer_values):
        span_claim = replace(claim, subjects=(span,))
        for entity in query_candidate_entities(connection, span_claim, per_query=32):
            selected.setdefault(entity["qid"], None)
    return list(selected)


def structured_facts(
    connection: sqlite3.Connection,
    qids: Iterable[str],
    *,
    reverse: bool,
) -> list[str]:
    ids = node_ids(connection, qids)
    if not ids:
        return []
    rows = query_facts(
        connection,
        subject_ids=ids.values() if not reverse else (),
        object_ids=ids.values() if reverse else (),
        predicates=TARGET_PREDICATES,
        years=set(),
        limit=512,
    )
    # Do not score the subject title: on identity questions it can equal the
    # teacher's answer even when the attached fact is completely irrelevant.
    return [render_fact(row) for row in rows]


def card_facts(connection: sqlite3.Connection, qids: Iterable[str]) -> list[str]:
    qids = list(dict.fromkeys(qids))
    if not qids:
        return []
    placeholders = ",".join("?" for _ in qids)
    rows = connection.execute(
        f"SELECT label,facts FROM entity WHERE qid IN ({placeholders})", qids
    ).fetchall()
    return [
        fact
        for _label, raw_facts in rows
        for fact in raw_facts.split("; ")
        if fact
    ]


def best_fact(source: dict[str, Any], facts: Iterable[str]) -> dict[str, Any]:
    scored = [
        (fact, score_record(source, fact.partition(": ")[2] or fact))
        for fact in dict.fromkeys(facts)
    ]
    if not scored:
        return {
            "facts": 0, "fact": "", "novel_target_recall": 0.0,
            "evidence_precision": 0.0, "summary_recall": 0.0,
            "novel_hit": False, "high_precision_hit": False,
        }
    fact, metric = max(
        scored,
        key=lambda item: (
            item[1]["novel_target_recall"], item[1]["evidence_precision"],
            item[1]["summary_recall"], -len(item[0]),
        ),
    )
    return {
        "facts": len(scored), "fact": fact, **metric,
        "novel_hit": metric["novel_target_recall"] > 0,
        "high_precision_hit": (
            metric["novel_target_recall"] > 0 and metric["evidence_precision"] >= 0.25
        ),
    }


def summarize(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    values = [row[condition] for row in rows]
    return {
        "rows": len(values),
        "rows_with_facts": sum(value["facts"] > 0 for value in values),
        "novel_hit_rows": sum(value["novel_hit"] for value in values),
        "high_precision_hit_rows": sum(value["high_precision_hit"] for value in values),
        "mean_best_novel_recall": statistics.fmean(
            value["novel_target_recall"] for value in values
        ) if values else 0.0,
        "mean_best_precision": statistics.fmean(
            value["evidence_precision"] for value in values if value["facts"]
        ) if any(value["facts"] for value in values) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--entity-database", type=Path, required=True)
    parser.add_argument("--relation-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    teachers = {
        (row["dataset"], int(row["index"])): row
        for row in load(args.teacher_cache)
        if row.get("reasoning_summary") and not row.get("parse_error")
        and "varied-deception" in str(row.get("dataset", ""))
    }
    entity = sqlite3.connect(f"file:{args.entity_database.resolve()}?mode=ro", uri=True)
    relation = sqlite3.connect(f"file:{args.relation_database.resolve()}?mode=ro", uri=True)
    relation.row_factory = sqlite3.Row
    rows = []
    records = load(args.input)
    if args.limit is not None:
        records = records[:args.limit]
    for number, record in enumerate(records, 1):
        key = (record["dataset"], int(record["index"]))
        teacher = teachers.get(key)
        if not teacher:
            continue
        source = {
            "conversation": teacher["student_prompt"],
            "reasoning_summary": teacher["reasoning_summary"],
        }
        current_qids = list(record.get("qids", []))
        all_qids = broad_qids(entity, record["conversation"])
        current_forward = structured_facts(relation, current_qids, reverse=False)
        broad_forward = structured_facts(relation, all_qids, reverse=False)
        broad_bidirectional = [
            *broad_forward, *structured_facts(relation, all_qids, reverse=True),
        ]
        full_cards = card_facts(entity, all_qids)
        rows.append({
            "dataset": record["dataset"], "index": record["index"],
            "question": extract_claim_query(record["conversation"]).question,
            "current_qids": current_qids, "broad_qids": all_qids,
            "current_structured": best_fact(source, current_forward),
            "broad_structured": best_fact(source, broad_forward),
            "bidirectional_structured": best_fact(source, broad_bidirectional),
            "full_cards": best_fact(source, full_cards),
        })
        if number % 100 == 0:
            print(f"processed {number}/{len(records)}", flush=True)
    entity.close()
    relation.close()
    conditions = (
        "current_structured", "broad_structured", "bidirectional_structured", "full_cards",
    )
    report = {
        "conditions": {condition: summarize(rows, condition) for condition in conditions},
        "incremental_high_precision_hits": {
            condition: sum(
                row[condition]["high_precision_hit"]
                and not row["current_structured"]["high_precision_hit"]
                for row in rows
            )
            for condition in conditions[1:]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
