#!/usr/bin/env python3
"""Evaluate a frozen local Wikidata FTS index on cached relevance rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any
import re

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.wikidata_rag.evaluate_retrieval import (
    STOPWORDS,
    TOKEN_RE,
    evidence_text,
    extract_conversation,
    score_record,
    search_queries,
    write_report,
)


def fts_expression(text: str, *, phrase: bool, max_tokens: int = 12) -> str:
    terms = []
    for match in TOKEN_RE.findall(text):
        term = match.lower().replace("’", "'")
        if len(term) >= 3 and term not in STOPWORDS and term not in terms:
            terms.append(term)
        if len(terms) == max_tokens:
            break
    if not terms:
        return ""
    escaped = [term.replace('"', '""') for term in terms]
    if phrase and len(escaped) > 1:
        return '"' + " ".join(escaped) + '"'
    return " OR ".join(f'"{term}"' for term in escaped)


def query_index(
    connection: sqlite3.Connection,
    conversation: str,
    limit: int,
) -> list[dict[str, Any]]:
    queries = search_queries(conversation)
    marked = []
    for pattern in (r"\*\*([^*]{2,100})\*\*", r"(?<!\*)\*([^*]{2,100})\*(?!\*)", r'["“]([^"”]{2,100})["”]'):
        marked.extend(re.findall(pattern, conversation))
    candidates = list(dict.fromkeys(marked + queries[2:]))

    selected: dict[str, dict[str, Any]] = {}
    sql = """
        SELECT e.qid, e.label, e.aliases, e.description, e.facts,
               bm25(entity_fts, 8.0, 5.0, 1.0, 1.0) AS rank
        FROM entity_fts
        JOIN entity AS e ON e.rowid = entity_fts.rowid
        WHERE entity_fts MATCH ?
        ORDER BY rank, e.popularity DESC
        LIMIT ?
    """
    def run(expression: str, count: int) -> list[tuple[Any, ...]]:
        try:
            return connection.execute(sql, (expression, count)).fetchall()
        except sqlite3.OperationalError:
            return []

    # First accept only genuine label/alias matches for explicit marked or named
    # spans. This prevents a phrase hit in an unrelated description from filling
    # all result slots.
    for candidate in candidates:
        expression = fts_expression(candidate, phrase=True)
        if not expression:
            continue
        candidate_terms = " ".join(match.lower() for match in TOKEN_RE.findall(candidate))
        rows = run(expression, limit * 3)
        for row in rows:
            names = " ".join(TOKEN_RE.findall(f"{row[1]} {row[2]}".lower()))
            if candidate_terms not in names and names not in candidate_terms:
                continue
            if row[0] in selected:
                continue
            selected[row[0]] = {
                "qid": row[0], "title": row[1], "label": row[1],
                "aliases": row[2].split("; ") if row[2] else [],
                "description": row[3],
                "facts": row[4].split("; ") if row[4] else [],
                "rank": row[5],
            }
            if len(selected) == limit:
                return list(selected.values())

    # Fill remaining slots from the rarest terms in the full exchange. Rare
    # terms are substantially safer than a flat OR over generic question words.
    all_terms = []
    for query in queries[:2]:
        for match in TOKEN_RE.findall(query):
            term = match.lower().replace("’", "'")
            if len(term) >= 3 and term not in STOPWORDS and term not in all_terms:
                all_terms.append(term)
    if all_terms:
        placeholders = ",".join("?" for _ in all_terms)
        frequencies = dict(connection.execute(
            f"SELECT term, doc FROM temp.fts_vocab WHERE term IN ({placeholders})",
            all_terms,
        ))
        rare = sorted(
            (term for term in all_terms if term in frequencies),
            key=lambda term: (frequencies[term], -len(term), all_terms.index(term)),
        )[:8]
        expression = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in rare)
        for row in (run(expression, limit * 3) if expression else []):
            if row[0] in selected:
                continue
            selected[row[0]] = {
                "qid": row[0], "title": row[1], "label": row[1],
                "aliases": row[2].split("; ") if row[2] else [],
                "description": row[3],
                "facts": row[4].split("; ") if row[4] else [],
                "rank": row[5],
            }
            if len(selected) == limit:
                break
    return list(selected.values())


def add_within_label_null(rows: list[dict[str, Any]]) -> None:
    alternatives = {}
    for label in (0, 1):
        group = [row for row in rows if row["label"] == label]
        evidences = [evidence_text(row["entities"]) for row in group]
        shifted = evidences[1:] + evidences[:1]
        for row, evidence in zip(group, shifted, strict=True):
            alternatives[(row["dataset"], row["index"])] = evidence
    for row in rows:
        row["shuffled_novel_target_recall"] = score_record(
            row, alternatives[(row["dataset"], row["index"])]
        )["novel_target_recall"]


def prepare_source(
    raw_source: list[dict[str, Any]], dataset_contains: str
) -> list[dict[str, Any]]:
    """Normalize pilot rows or parsed teacher-cache rows for evaluation."""
    source = []
    for item in raw_source:
        if dataset_contains and dataset_contains not in str(item.get("dataset", "")):
            continue
        if item.get("parse_error", False):
            continue
        if "conversation" not in item:
            if not item.get("student_prompt") or not item.get("reasoning_summary"):
                continue
            item = {**item, "conversation": extract_conversation(item["student_prompt"])}
        source.append(item)
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit-results", type=int, default=3)
    parser.add_argument("--dataset-contains", default="varied-deception")
    args = parser.parse_args()

    raw_source = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    source = prepare_source(raw_source, args.dataset_contains)
    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    connection.execute(
        "CREATE VIRTUAL TABLE temp.fts_vocab USING fts5vocab(main, entity_fts, 'row')"
    )
    rows = []
    for number, item in enumerate(source, 1):
        row = {
            key: item[key]
            for key in ("dataset", "index", "label", "conversation", "reasoning_summary")
        }
        row["entities"] = query_index(connection, row["conversation"], args.limit_results)
        row["error"] = None
        row.update(score_record(row, evidence_text(row["entities"])))
        rows.append(row)
        if number % 20 == 0:
            print(f"queried {number}/{len(source)}", flush=True)
    connection.close()
    add_within_label_null(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    write_report(args.report, rows)
    print(f"wrote {args.output} and {args.report}")


if __name__ == "__main__":
    main()
