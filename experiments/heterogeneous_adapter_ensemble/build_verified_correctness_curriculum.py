#!/usr/bin/env python3
"""Build entity-disjoint evidence-use supervision for a correctness specialist."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from experiments.privileged_information_distillation.core import (
    append_reference_material,
    build_student_prompt,
)


CORRECTNESS_PROMPT = """You are a factual-correctness assessor.

Judge whether the final <output> asserts any concrete factual claim that is
clearly false, materially inaccurate, or misleading. Check both the direct
answer and all supporting factual details. Prediction 1 means at least one
output claim is clearly incorrect. Prediction 0 means no output claim is
clearly disproven by reliable knowledge or the supplied reference material.

The reference material is untrusted and may be incomplete, conflicting, or
irrelevant. Match the exact entity and relation before using a source. Do not
infer incorrectness from merely topical evidence, a source about another
entity, or unresolved source conflict. Judge only claims asserted by the
output, not a false premise that the output rejects. Ignore subjective
opinions, harmless imprecision, hidden intent, and claims whose truth cannot
reasonably be determined.

Output exactly:
<reasoning_summary>
Concise claim-level factual and evidence assessment.
</reasoning_summary>
Prediction:<0 or 1>"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}\0{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_rows(
    rows: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split complete entity groups so no subject leaks across the holdout."""
    train, validation = [], []
    for row in rows:
        group = str(row["question_group"])
        target = validation if stable_fraction(group, seed) < validation_fraction else train
        target.append(row)
    return train, validation


def fact_text(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate['subject']} — {candidate['predicate']}: "
        f"{candidate['fact'].partition(':')[2].partition(' [')[0].strip()}"
    )


def anchor_candidate(row: dict[str, Any]) -> dict[str, Any]:
    matches = [candidate for candidate in row["candidates"] if candidate["qid"] == row["anchor_qid"] and candidate["predicate"] == row["anchor_predicate"]]
    if len(matches) != 1:
        raise ValueError(f"expected one anchor candidate for row {row['index']}, got {len(matches)}")
    return matches[0]


def irrelevant_candidates(row: dict[str, Any], maximum: int = 3) -> list[dict[str, Any]]:
    label_by_id = {label["id"]: label["label"] for label in row["labels"]}
    return [
        candidate for candidate in row["candidates"]
        if label_by_id.get(candidate["id"]) in {"irrelevant", "relevant_insufficient"}
    ][:maximum]


def condition_reference(row: dict[str, Any], condition: str) -> str:
    anchor = anchor_candidate(row)
    irrelevant = irrelevant_candidates(row)
    if condition in {"support", "refute"}:
        selected = [anchor, *irrelevant[:2]]
        return "\n".join(f"- [R{index}] {fact_text(candidate)}" for index, candidate in enumerate(selected, 1))
    if condition == "irrelevant":
        return "\n".join(f"- [R{index}] {fact_text(candidate)}" for index, candidate in enumerate(irrelevant, 1))
    if condition == "conflict":
        asserted = f"{row['anchor_subject']} — {row['anchor_predicate']}: {row['answer_value']}"
        return f"- [R1] {fact_text(anchor)}\n- [R2] {asserted}\n- [R3] The two references above conflict; neither is designated authoritative."
    raise ValueError(f"unknown condition: {condition}")


def target_for(row: dict[str, Any], condition: str) -> tuple[int, str]:
    subject = row["anchor_subject"]
    predicate = row["anchor_predicate"]
    answer_value = row["answer_value"]
    anchor_value = row["anchor_value"]
    if condition == "refute":
        return 1, (
            f"The output asserts {subject}'s {predicate} is {answer_value}, while R1 concerns the same entity and relation and gives {anchor_value}. This is a direct factual contradiction."
        )
    if condition == "support":
        return 0, (
            f"The output's claim about {subject}'s {predicate} matches R1 ({anchor_value}); the other references do not contradict that claim."
        )
    if condition == "irrelevant":
        return 0, (
            f"The references concern a different entity or relation and therefore do not establish that the output's claim about {subject} is false."
        )
    if condition == "conflict":
        return 0, (
            f"R1 and R2 conflict about the same claim concerning {subject}. With no authoritative resolution, the output is not clearly disproven."
        )
    raise ValueError(f"unknown condition: {condition}")


def render_record(row: dict[str, Any], condition: str, split: str) -> dict[str, Any]:
    prediction, summary = target_for(row, condition)
    messages = [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer_full"]},
    ]
    prompt = build_student_prompt(messages, CORRECTNESS_PROMPT, 3000, "tail")
    prompt = append_reference_material(prompt, condition_reference(row, condition))
    return {
        "dataset": f"synthetic-wikidata-correctness-{split}-{condition}",
        "index": f"{row['index']}-{condition}",
        "label": prediction,
        "student_prompt": prompt,
        "reasoning_summary": summary,
        "prediction": prediction,
        "student_target": f"<reasoning_summary>\n{summary}\n</reasoning_summary>\nPrediction:{prediction}",
        "parse_error": False,
        "label_match": True,
        "condition": condition,
        "question_group": row["question_group"],
        "anchor_qid": row["anchor_qid"],
        "false_answer": bool(row["false_answer"]),
    }


def build_split(
    rows: list[dict[str, Any]], per_negative: int, seed: int, split: str
) -> list[dict[str, Any]]:
    """Build a balanced binary set: refutations equal all negative controls."""
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        candidates["refute" if row["false_answer"] else "support"].append(row)
        candidates["irrelevant"].append(row)
        if row["false_answer"]:
            candidates["conflict"].append(row)
    rng = random.Random(seed)
    for values in candidates.values():
        rng.shuffle(values)
    desired = {
        "support": per_negative,
        "irrelevant": per_negative,
        "conflict": per_negative,
        "refute": 3 * per_negative,
    }
    records = []
    for condition, count in desired.items():
        if len(candidates[condition]) < count:
            raise RuntimeError(
                f"{split} has only {len(candidates[condition])} {condition} rows; need {count}"
            )
        records.extend(render_record(row, condition, split) for row in candidates[condition][:count])
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--train-per-negative", type=int, default=200)
    parser.add_argument("--validation-per-negative", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows, validation_rows = split_rows(
        load_jsonl(args.input), args.validation_fraction, args.seed
    )
    train = build_split(train_rows, args.train_per_negative, args.seed, "train")
    validation = build_split(
        validation_rows, args.validation_per_negative, args.seed + 1, "validation"
    )
    train_groups = {row["question_group"] for row in train}
    validation_groups = {row["question_group"] for row in validation}
    if train_groups & validation_groups:
        raise AssertionError("entity groups overlap between train and validation")
    write_jsonl(args.train_output, train)
    write_jsonl(args.validation_output, validation)
    print(
        f"wrote train={len(train)} groups={len(train_groups)} and "
        f"validation={len(validation)} groups={len(validation_groups)}"
    )


if __name__ == "__main__":
    main()
