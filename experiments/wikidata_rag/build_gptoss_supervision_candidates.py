#!/usr/bin/env python3
"""Build bounded, label-blind Wikidata candidate sets for GPT-OSS supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from experiments.wikidata_rag.claim_retrieval import (
    content_tokens,
    extract_claim_query,
    fact_predicate,
    normalize,
    normalize_name,
    parse_final_exchange,
    split_facts,
)
from experiments.wikidata_rag.evaluate_relation_oracle import broad_qids


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def question_group(question: str) -> str:
    """Stable exact-question group used to prevent template leakage."""
    return hashlib.sha1(normalize(question).encode()).hexdigest()[:16]


def entity_rows(connection: sqlite3.Connection, qids: list[str]) -> list[dict[str, Any]]:
    if not qids:
        return []
    placeholders = ",".join("?" for _ in qids)
    rows = connection.execute(
        f"SELECT qid,label,facts,popularity FROM entity WHERE qid IN ({placeholders})",
        qids,
    ).fetchall()
    return [
        {"qid": qid, "subject": label, "facts": split_facts(facts), "popularity": popularity}
        for qid, label, facts, popularity in rows
    ]


def candidate_score(
    question: str,
    answer: str,
    subjects: tuple[str, ...],
    routed_predicates: set[str],
    entity: dict[str, Any],
    fact: str,
) -> tuple[int, int, int, int]:
    predicate = fact_predicate(fact)
    value = fact.partition(":")[2]
    q_tokens = content_tokens(question)
    a_tokens = content_tokens(answer)
    value_tokens = content_tokens(value)
    subject_norm = normalize_name(entity["subject"])
    subject_match = int(any(subject_norm == normalize_name(span) for span in subjects))
    return (
        100 * int(predicate in routed_predicates)
        + 40 * subject_match
        + 25 * int(bool(normalize(value)) and normalize(value) in normalize(answer))
        + 8 * len(value_tokens & a_tokens)
        + 3 * len(value_tokens & q_tokens),
        subject_match,
        len(value_tokens & a_tokens),
        int(entity["popularity"] or 0),
    )


def select_candidates(
    connection: sqlite3.Connection,
    conversation: str,
    *,
    limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    claim = extract_claim_query(conversation)
    _, answer_full = parse_final_exchange(conversation)
    qids = broad_qids(connection, conversation)
    candidates: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for entity in entity_rows(connection, qids):
        grouped: dict[str, list[str]] = {}
        for fact in entity["facts"]:
            predicate, separator, value = fact.partition(":")
            if separator:
                grouped.setdefault(predicate.strip().lower(), []).append(value.strip())
        for predicate, values in grouped.items():
            rendered = f"{predicate}: {' | '.join(values[:6])}"
            key = (str(entity["qid"]), rendered)
            if key in seen:
                continue
            seen.add(key)
            item = {
                "qid": str(entity["qid"]),
                "subject": str(entity["subject"]),
                "predicate": predicate,
                "fact": rendered[:500],
                "popularity": int(entity["popularity"] or 0),
            }
            candidates.append((candidate_score(
                claim.question, answer_full, claim.subjects, set(claim.predicates),
                entity, rendered,
            ), item))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [item for _, item in candidates[:limit]]
    for index, item in enumerate(selected):
        item["id"] = f"C{index:02d}"
    return {
        "question": claim.question,
        "answer": claim.answer,
        "answer_full": answer_full,
        "subjects": list(claim.subjects),
        "rule_predicates": list(claim.predicates),
        "question_group": question_group(claim.question),
        "linked_qids": qids,
    }, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--entity-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-per-row", type=int, default=16)
    args = parser.parse_args()

    connection = sqlite3.connect(f"file:{args.entity_database.resolve()}?mode=ro", uri=True)
    source = load(args.input)
    output = []
    for number, row in enumerate(source, 1):
        claim, candidates = select_candidates(
            connection, row["conversation"], limit=args.candidates_per_row
        )
        output.append({
            "dataset": row["dataset"], "index": row["index"],
            "conversation": row["conversation"], **claim, "candidates": candidates,
            "currently_covered": bool(row.get("real_passages")),
        })
        if number % 250 == 0:
            print(f"processed {number}/{len(source)}", flush=True)
    connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )
    print(f"wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
