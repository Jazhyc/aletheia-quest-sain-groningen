#!/usr/bin/env python3
"""Retrieve real and deterministic shuffled Wikidata evidence for a public split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from experiments.privileged_information_distillation.core import format_example
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import (
    load_labels,
    load_split_config,
)
from experiments.wikidata_rag.evaluate_offline_index import query_index
from experiments.wikidata_rag.evaluate_retrieval import tokens


def entity_passages(entities: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Render compact cards in the cache format consumed by the student evaluator."""
    passages = []
    for entity in entities:
        details = "; ".join(
            part
            for part in (
                entity.get("description", ""),
                "; ".join(entity.get("facts", [])[:8]),
            )
            if part
        )
        passages.append({
            "title": f"{entity.get('label', '')} ({entity.get('qid', '')})",
            "text": details,
        })
    return passages


def passage_text(passages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- {passage.get('title', '')}: {passage.get('text', '')}"
        for passage in passages
    )


def choose_group_rotation(
    recipients: list[dict[str, Any]], donors: list[dict[str, Any]]
) -> int:
    """Choose a bijective cross-group rotation with no repeated row index."""
    if len(recipients) != len(donors) or not recipients:
        raise ValueError("recipient and donor groups must have the same nonzero size")
    best: tuple[tuple[int, float, float], int] | None = None
    for shift in range(len(donors)):
        repeated_indexes = 0
        overlaps = []
        length_errors = []
        for position, recipient in enumerate(recipients):
            donor = donors[(position + shift) % len(donors)]
            repeated_indexes += int(recipient["index"] == donor["index"])
            conversation_terms = tokens(recipient["conversation"])
            donor_terms = tokens(passage_text(donor["real_passages"]))
            union = conversation_terms | donor_terms
            overlaps.append(len(conversation_terms & donor_terms) / max(1, len(union)))
            real_length = max(1, len(passage_text(recipient["real_passages"])))
            donor_length = max(1, len(passage_text(donor["real_passages"])))
            length_errors.append(abs(math.log(donor_length / real_length)))
        score = (
            repeated_indexes,
            sum(overlaps) / len(overlaps),
            sum(length_errors) / len(length_errors),
        )
        if best is None or score < best[0]:
            best = (score, shift)
    assert best is not None
    return best[1]


def add_shuffled_passages(records: list[dict[str, Any]], seed: int) -> list[int]:
    """Attach one-to-one shuffled cards and donor provenance to each record."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["dataset"], []).append(record)
    if len(grouped) < 2:
        raise ValueError("at least two dataset groups are required for shuffled evidence")
    sizes = {len(group) for group in grouped.values()}
    if len(sizes) != 1:
        raise ValueError(f"dataset groups must have equal sizes, got {sorted(sizes)}")
    rng = random.Random(seed)
    dataset_order = sorted(grouped)
    rng.shuffle(dataset_order)
    rotations = []
    for position, dataset in enumerate(dataset_order):
        donor_dataset = dataset_order[(position + 1) % len(dataset_order)]
        recipients = sorted(grouped[dataset], key=lambda row: str(row["index"]))
        donors = sorted(grouped[donor_dataset], key=lambda row: str(row["index"]))
        rotation = choose_group_rotation(recipients, donors)
        rotations.append(rotation)
        for row_number, recipient in enumerate(recipients):
            donor = donors[(row_number + rotation) % len(donors)]
            recipient["shuffled_passages"] = donor["real_passages"]
            recipient["shuffled_from"] = {
                "dataset": donor["dataset"],
                "index": donor["index"],
            }
    if any(
        row["dataset"] == row["shuffled_from"]["dataset"]
        or row["index"] == row["shuffled_from"]["index"]
        for row in records
    ):
        raise RuntimeError("could not construct a cross-dataset, cross-index evidence shuffle")
    return rotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--scenario", default="varied-deception")
    parser.add_argument("--limit-results", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    connection.execute(
        "CREATE VIRTUAL TABLE temp.fts_vocab USING fts5vocab(main, entity_fts, 'row')"
    )
    records = []
    for config in load_split_config(args.splits_dir / f"dry.{args.split}.yaml", ROOT):
        if args.scenario not in config.name:
            continue
        labels = load_labels(config)
        wanted = set(labels["index"].tolist())
        dataset = load_dataset(config.name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        for row in dataset:
            if row["index"] not in wanted:
                continue
            conversation, _ = format_example(row["messages"], 3000, "tail")
            entities = query_index(connection, conversation, args.limit_results)
            records.append({
                "dataset": config.name,
                "index": row["index"],
                "conversation": conversation,
                "real_passages": entity_passages(entities),
                "qids": [entity["qid"] for entity in entities],
            })
    connection.close()
    rotations = add_shuffled_passages(records, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    print(
        f"wrote {len(records)} rows to {args.output}; rotations={rotations}; "
        f"real_cards={sum(len(row['real_passages']) for row in records)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
