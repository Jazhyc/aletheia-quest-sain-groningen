from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.heterogeneous_adapter_ensemble.build_verified_correctness_curriculum import (
    CORRECTNESS_PROMPT,
    condition_reference,
    render_record,
    split_rows,
)


def sample_row(*, false_answer: bool) -> dict:
    answer = "Lyon" if false_answer else "Paris"
    return {
        "index": 1,
        "question": "What is the capital of France?",
        "answer": f"The capital of France is {answer}.",
        "answer_full": f"The capital of France is {answer}.",
        "question_group": "france",
        "anchor_qid": "Q142",
        "anchor_subject": "France",
        "anchor_predicate": "capital",
        "anchor_value": "Paris",
        "answer_value": answer,
        "false_answer": false_answer,
        "candidates": [
            {"id": "C00", "qid": "Q142", "subject": "France", "predicate": "capital", "fact": "capital: Paris"},
            {"id": "C01", "qid": "Q90", "subject": "Paris", "predicate": "country", "fact": "country: France"},
            {"id": "C02", "qid": "Q183", "subject": "Germany", "predicate": "capital", "fact": "capital: Berlin"},
        ],
        "labels": [
            {"id": "C00", "label": "contradicts" if false_answer else "supports"},
            {"id": "C01", "label": "relevant_insufficient"},
            {"id": "C02", "label": "irrelevant"},
        ],
    }


def test_rendered_conditions_have_expected_verified_labels() -> None:
    assert render_record(sample_row(false_answer=False), "support", "train")["label"] == 0
    assert render_record(sample_row(false_answer=True), "refute", "train")["label"] == 1
    assert render_record(sample_row(false_answer=True), "irrelevant", "train")["label"] == 0
    assert render_record(sample_row(false_answer=True), "conflict", "train")["label"] == 0


def test_references_enforce_entity_relation_alignment() -> None:
    row = sample_row(false_answer=True)
    irrelevant = condition_reference(row, "irrelevant")
    assert "France — capital: Paris" not in irrelevant
    conflict = condition_reference(row, "conflict")
    assert "France — capital: Paris" in conflict
    assert "France — capital: Lyon" in conflict


def test_group_split_is_disjoint_and_stable() -> None:
    rows = [dict(sample_row(false_answer=False), question_group=f"g{i}") for i in range(50)]
    train_a, validation_a = split_rows(rows, 0.2, 42)
    train_b, validation_b = split_rows(rows, 0.2, 42)
    assert [row["question_group"] for row in train_a] == [row["question_group"] for row in train_b]
    assert [row["question_group"] for row in validation_a] == [row["question_group"] for row in validation_b]
    assert {row["question_group"] for row in train_a}.isdisjoint(
        {row["question_group"] for row in validation_a}
    )


def test_training_config_uses_the_curriculum_prompt_and_rank_one() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/pid_heterogeneous_verified_correctness_rank1_v1.yaml").read_text()
    )
    student = config["student"]
    assert student["prompt"].strip() == CORRECTNESS_PROMPT.strip()
    assert student["selection_manifest"] is None
    assert any(
        "curriculum/train.jsonl" in source["artifact"]
        for source in student["teacher_sources"]
    )
