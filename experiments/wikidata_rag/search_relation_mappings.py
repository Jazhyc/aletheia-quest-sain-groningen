#!/usr/bin/env python3
"""Search predicate mappings by teacher-summary relevance, never class labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Any

from experiments.wikidata_rag.build_relation_index import TARGET_PREDICATES
from experiments.wikidata_rag.evaluate_retrieval import score_record
from experiments.wikidata_rag.relation_retrieval import node_ids, query_facts, render_fact


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_for(dataset: str, index: int) -> str:
    digest = hashlib.sha1(f"{dataset}:{index}".encode()).digest()[0]
    return "selection" if digest % 2 == 0 else "confirmation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation-cache", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    teachers = {
        (row["dataset"], int(row["index"])): row for row in load(args.teacher_cache)
        if row.get("reasoning_summary") and not row.get("parse_error")
    }
    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    scores: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for row in load(args.relation_cache):
        key = (row["dataset"], int(row["index"]))
        teacher = teachers.get(key)
        relations = row.get("claim", {}).get("relations", [])
        if not teacher or not relations or not row.get("qids"):
            continue
        ids = node_ids(connection, row["qids"])
        if not ids:
            continue
        source = {"conversation": teacher["student_prompt"],
                  "reasoning_summary": teacher["reasoning_summary"]}
        split = split_for(*key)
        for predicate in TARGET_PREDICATES:
            facts = query_facts(
                connection, subject_ids=ids.values(), predicates=(predicate,), years=set(), limit=8
            )
            if not facts:
                continue
            metric = score_record(source, "; ".join(render_fact(fact) for fact in facts))
            for relation in relations:
                scores[(relation, predicate, split)].append(metric)
    report: dict[str, Any] = {"relations": {}}
    relation_names = sorted({relation for relation, _, _ in scores})
    for relation in relation_names:
        candidates = []
        for predicate in TARGET_PREDICATES:
            split_rows = {}
            for split in ("selection", "confirmation"):
                rows = scores.get((relation, predicate, split), [])
                split_rows[split] = {
                    "covered": len(rows),
                    "novel_hits": sum(row["novel_target_recall"] > 0 for row in rows),
                    "precision": statistics.fmean(row["evidence_precision"] for row in rows) if rows else 0.0,
                    "novel_recall": statistics.fmean(row["novel_target_recall"] for row in rows) if rows else 0.0,
                }
            if split_rows["selection"]["covered"]:
                candidates.append({"predicate": predicate, **split_rows})
        candidates.sort(key=lambda row: (
            -min(row["selection"]["novel_hits"], row["confirmation"]["novel_hits"]),
            -(row["selection"]["precision"] + row["confirmation"]["precision"]),
            -(row["selection"]["covered"] + row["confirmation"]["covered"]),
        ))
        report["relations"][relation] = candidates[:10]
    connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
