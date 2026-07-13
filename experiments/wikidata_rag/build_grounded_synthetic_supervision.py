#!/usr/bin/env python3
"""Create fact-grounded QA/ranking examples with mechanically known labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
from typing import Any

from experiments.wikidata_rag.claim_retrieval import fact_predicate, split_facts


TEMPLATES: dict[str, tuple[str, str]] = {
    "country": ("Which country is {subject} in?", "{subject} is in {value}."),
    "country of origin": ("What is {subject}'s country of origin?", "{subject} originates from {value}."),
    "capital": ("What is the capital of {subject}?", "The capital of {subject} is {value}."),
    "place of birth": ("Where was {subject} born?", "{subject} was born in {value}."),
    "place of death": ("Where did {subject} die?", "{subject} died in {value}."),
    "date of birth": ("When was {subject} born?", "{subject} was born on {value}."),
    "date of death": ("When did {subject} die?", "{subject} died on {value}."),
    "inception": ("When was {subject} established?", "{subject} was established on {value}."),
    "publication date": ("When was {subject} published or released?", "{subject} was published or released on {value}."),
    "author": ("Who wrote {subject}?", "{subject} was written by {value}."),
    "director": ("Who directed {subject}?", "{subject} was directed by {value}."),
    "screenwriter": ("Who wrote the screenplay for {subject}?", "The screenplay for {subject} was written by {value}."),
    "composer": ("Who composed {subject}?", "{subject} was composed by {value}."),
    "creator": ("Who created {subject}?", "{subject} was created by {value}."),
    "performer": ("Who performed {subject}?", "{subject} was performed by {value}."),
    "headquarters": ("Where is {subject} headquartered?", "{subject} is headquartered in {value}."),
    "official language": ("What is the official language of {subject}?", "The official language of {subject} is {value}."),
    "genre": ("What genre is {subject}?", "{subject} belongs to the {value} genre."),
    "occupation": ("What is {subject}'s occupation?", "{subject} works as a {value}."),
    "instance of": ("What type of thing is {subject}?", "{subject} is a {value}."),
    "subclass of": ("What broader type does {subject} belong to?", "{subject} is a type of {value}."),
    "position held": ("Which position did {subject} hold?", "{subject} held the position of {value}."),
}

FUNCTIONAL = {
    "capital", "country", "country of origin", "date of birth", "date of death",
    "director", "headquarters", "inception", "place of birth", "place of death",
    "publication date", "screenwriter",
}


def clean_value(fact: str) -> str:
    return fact.partition(":")[2].partition(" [")[0].strip()


def usable_value(value: str) -> bool:
    return (
        2 <= len(value) <= 100
        and " | " not in value
        and value.lower() not in {"unknown", "none"}
        and not re.search(r"\b(?:somevalue|novalue)\b", value, re.I)
    )


def entity_group(qid: str) -> str:
    return hashlib.sha1(qid.encode()).hexdigest()[:16]


def load_cards(connection: sqlite3.Connection, maximum: int) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT qid,label,facts,popularity FROM entity ORDER BY popularity DESC LIMIT ?",
        (maximum,),
    ).fetchall()
    cards = []
    for qid, label, raw_facts, popularity in rows:
        facts = [
            fact for fact in split_facts(raw_facts)
            if fact_predicate(fact) in TEMPLATES and usable_value(clean_value(fact))
        ]
        if facts:
            cards.append({
                "qid": str(qid), "subject": str(label), "facts": facts,
                "popularity": int(popularity or 0),
            })
    return cards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--maximum-cards", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    connection = sqlite3.connect(f"file:{args.entity_database.resolve()}?mode=ro", uri=True)
    cards = load_cards(connection, args.maximum_cards)
    connection.close()
    by_predicate: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for card in cards:
        for fact in card["facts"]:
            by_predicate[fact_predicate(fact)].append((card, fact))
    anchors = [(card, fact) for card in cards for fact in card["facts"]]
    rng.shuffle(anchors)

    output = []
    used: set[tuple[str, str]] = set()
    for anchor, fact in anchors:
        predicate = fact_predicate(fact)
        key = (anchor["qid"], predicate)
        if key in used:
            continue
        used.add(key)
        correct_value = clean_value(fact)
        contradictory = predicate in FUNCTIONAL and len(by_predicate[predicate]) > 1
        make_false = contradictory and len(output) % 2 == 1
        answer_value = correct_value
        if make_false:
            alternative = None
            for _ in range(50):
                other_card, other_fact = rng.choice(by_predicate[predicate])
                other_value = clean_value(other_fact)
                if other_card["qid"] != anchor["qid"] and other_value != correct_value:
                    alternative = other_value
                    break
            if alternative is None:
                make_false = False
            else:
                answer_value = alternative
        question_template, answer_template = TEMPLATES[predicate]
        question = question_template.format(subject=anchor["subject"])
        answer = answer_template.format(subject=anchor["subject"], value=answer_value)
        candidates = [{
            "id": "C00", "qid": anchor["qid"], "subject": anchor["subject"],
            "predicate": predicate, "fact": fact, "popularity": anchor["popularity"],
        }]
        labels = [{
            "id": "C00", "label": "contradicts" if make_false else "supports",
            "claim_quote": answer, "reason": "mechanically grounded anchor fact",
        }]
        wrong_relation = [value for value in anchor["facts"] if fact_predicate(value) != predicate]
        rng.shuffle(wrong_relation)
        for negative in wrong_relation[:2]:
            candidate_id = f"C{len(candidates):02d}"
            candidates.append({
                "id": candidate_id, "qid": anchor["qid"], "subject": anchor["subject"],
                "predicate": fact_predicate(negative), "fact": negative,
                "popularity": anchor["popularity"],
            })
            labels.append({
                "id": candidate_id, "label": "relevant_insufficient",
                "claim_quote": answer, "reason": "same entity, wrong requested relation",
            })
        same_predicate_pool = by_predicate[predicate]
        same_predicate_added = 0
        attempts = 0
        while same_predicate_added < 2 and attempts < 100:
            attempts += 1
            card, negative = rng.choice(same_predicate_pool)
            if card["qid"] == anchor["qid"]:
                continue
            candidate_id = f"C{len(candidates):02d}"
            candidates.append({
                "id": candidate_id, "qid": card["qid"], "subject": card["subject"],
                "predicate": predicate, "fact": negative,
                "popularity": card["popularity"],
            })
            labels.append({
                "id": candidate_id, "label": "irrelevant", "claim_quote": "NONE",
                "reason": "requested relation, different entity",
            })
            same_predicate_added += 1
        attempts = 0
        while len(candidates) < args.candidate_count and attempts < 10 * args.candidate_count:
            attempts += 1
            card, negative = rng.choice(anchors)
            if card["qid"] == anchor["qid"]:
                continue
            candidate_id = f"C{len(candidates):02d}"
            candidates.append({
                "id": candidate_id, "qid": card["qid"], "subject": card["subject"],
                "predicate": fact_predicate(negative), "fact": negative,
                "popularity": card["popularity"],
            })
            labels.append({
                "id": candidate_id, "label": "irrelevant", "claim_quote": "NONE",
                "reason": "different entity",
            })
        paired = list(zip(candidates, labels, strict=True))
        rng.shuffle(paired)
        candidates, labels = map(list, zip(*paired, strict=True))
        for index, (candidate, label) in enumerate(zip(candidates, labels, strict=True)):
            candidate["id"] = label["id"] = f"C{index:02d}"
        output.append({
            "dataset": "synthetic-wikidata", "index": len(output),
            "question": question, "answer": answer, "answer_full": answer,
            "question_group": entity_group(anchor["qid"]),
            "rule_predicates": [predicate], "linked_qids": [anchor["qid"]],
            "candidates": candidates, "labels": labels, "parse_error": False,
            "synthetic": True, "anchor_qid": anchor["qid"],
            "anchor_subject": anchor["subject"], "anchor_predicate": predicate,
            "anchor_value": correct_value,
            "answer_value": answer_value, "false_answer": make_false,
        })
        if len(output) == args.rows:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )
    print(f"wrote {len(output)} grounded synthetic rows from {len(cards)} cards")


if __name__ == "__main__":
    main()
