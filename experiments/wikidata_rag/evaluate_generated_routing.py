#!/usr/bin/env python3
"""Mechanically verify claim routing against templates generated from indexed facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from experiments.wikidata_rag.claim_retrieval import extract_claim_query


TEMPLATES = {
    "author": "Who wrote {subject}?", "composer": "Who composed {subject}?",
    "creator": "Who created {subject}?", "director": "Who directed {subject}?",
    "founded by": "Who founded {subject}?", "performer": "Who performed {subject}?",
    "screenwriter": "Who wrote the screenplay for {subject}?",
    "cast member": "Who starred in {subject}?", "voice actor": "Who voiced a character in {subject}?",
    "characters": "Which character appears in {subject}?",
    "country": "In which country is {subject}?", "country of origin": "What country is {subject} from?",
    "capital": "What is the capital of {subject}?",
    "located in": "In which city is {subject} located?", "location": "Where was {subject} held?",
    "place of burial": "Where was {subject} buried?",
    "sport": "Which sport did {subject} play?", "league": "Which league did {subject} play in?",
    "country for sport": "Which country did {subject} represent in sport?",
    "member of": "Which group was {subject} a member of?",
    "member of sports team": "Which team did {subject} play for?",
    "participant": "Who participated in {subject}?", "participant in": "Which event did {subject} participate in?",
    "organizer": "Who organized {subject}?", "winner": "Who won {subject}?",
    "conflict": "Which war did {subject} participate in?",
    "significant event": "Which major event involved {subject}?",
    "position held": "Which position did {subject} hold?",
    "head of government": "Who was the prime minister of {subject}?",
    "head of state": "Who was the head of state of {subject}?",
    "award received": "Which award did {subject} win?",
    "lyrics by": "Who wrote the lyrics for {subject}?",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    rows = []
    for predicate, template in TEMPLATES.items():
        # Predicate IDs are stable and declared by the builder; avoid storing a
        # redundant predicate table in the submission index.
        from experiments.wikidata_rag.build_relation_index import PREDICATE_IDS
        sample = connection.execute("""
            SELECT s.label,o.label FROM fact AS f JOIN node AS s ON s.id=f.subject
            LEFT JOIN node AS o ON o.id=f.object WHERE f.predicate=? LIMIT 1
        """, (PREDICATE_IDS[predicate],)).fetchone()
        if not sample:
            continue
        question = template.format(subject=sample[0])
        answer = sample[1] or "the stated value"
        claim = extract_claim_query(f"USER: {question}\nASSISTANT: {answer}.")
        rows.append({
            "predicate": predicate, "question": question,
            "routed_predicates": list(claim.predicates),
            "passed": predicate in claim.predicates,
        })
    connection.close()
    report = {
        "cases": len(rows), "passed": sum(row["passed"] for row in rows),
        "failed": [row for row in rows if not row["passed"]], "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
