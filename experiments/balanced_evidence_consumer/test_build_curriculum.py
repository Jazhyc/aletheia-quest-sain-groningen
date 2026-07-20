from __future__ import annotations

import json

from experiments.balanced_evidence_consumer.build_curriculum import (
    audit_variants,
    build_variants,
    contradiction_keys,
    insufficient_references,
    split_reference,
    with_evidence_assessment,
)
from experiments.privileged_information_distillation.core import (
    append_reference_material,
)


def record(dataset: str, index: int, label: int, reference: str = "") -> dict:
    return {
        "dataset": dataset,
        "index": index,
        "label": label,
        "student_prompt": append_reference_material("PROMPT", reference),
        "student_target": (
            "<reasoning_summary>\n"
            "x\n"
            "</reasoning_summary>\n"
            f"Prediction:{label}"
        ),
        "parse_error": False,
        "label_match": True,
    }


def test_split_reference_removes_preamble() -> None:
    base, reference = split_reference(
        append_reference_material("PROMPT", "- Paris: fact")
    )
    assert base == "PROMPT"
    assert reference == "- Paris: fact"


def test_with_evidence_assessment_preserves_prediction() -> None:
    updated = with_evidence_assessment(record("a", 1, 0), "No source is available.")
    assert "Evidence assessment: No source is available." in updated["student_target"]
    assert "Decision: x" in updated["student_target"]
    assert updated["student_target"].endswith("Prediction:0")


def test_audit_helpers_select_rejected_and_contradictory_rows(tmp_path) -> None:
    path = tmp_path / "audits.jsonl"
    rows = [
        {
            "dataset": "a",
            "index": 1,
            "label": 0,
            "parse_valid": True,
            "decision": "DECISIVE_CONTRADICTION",
            "selected_candidate": {"text": "decisive"},
            "proposition": "Paris is in Germany.",
            "candidates": [{"title": "Paris", "text": "Paris is a city."}],
            "candidate_assessments": [{"candidate_id": 1, "relation": "TOPICAL"}],
        }
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert "Paris is a city." in insufficient_references(path)[("a", 1)]
    assert contradiction_keys(path) == {0: [("a", 1)]}


def test_build_variants_balances_contradiction_strata() -> None:
    keys = [("a", 1), ("b", 2)]
    baseline = {
        key: record(key[0], key[1], position)
        for position, key in enumerate(keys)
    }
    real = {
        key: with_evidence_assessment(
            record(key[0], key[1], position, f"- real {position}"),
            "The real source is decisive.",
        )
        for position, key in enumerate(keys)
    }
    shuffled = {
        key: with_evidence_assessment(
            record(key[0], key[1], position, f"- shuffled {position}"),
            "The shuffled source is irrelevant.",
        )
        for position, key in enumerate(keys)
    }
    rows = build_variants(
        baseline,
        real,
        shuffled,
        {key: "- insufficient" for key in keys},
        {0: [keys[0]], 1: [keys[1]]},
    )
    variants = [row["evidence_variant"] for row in rows]
    assert variants.count("real_decisive") == 2
    assert variants.count("mixed_relevant_irrelevant") == 2
    assert variants.count("topical_or_insufficient") == 2
    assert variants.count("honest_ordinary_error_control") == 1
    assert variants.count("deceptive_contradiction_control") == 1
    report = audit_variants(rows)
    assert report["rows"] == 12
    assert report["labels"] == {"0": 6, "1": 6}
