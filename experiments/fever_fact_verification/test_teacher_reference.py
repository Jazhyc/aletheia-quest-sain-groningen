from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.build_teacher_reference import (
    build_references,
    limit_reference_passages,
    truncate_chars,
)


def test_build_references_uses_real_polar_evidence_only() -> None:
    base = {
        "dataset": "dataset",
        "index": 1,
        "claim_index": 0,
        "proposition": "Paris is in France.",
        "confidence": 0.9,
        "evidence": [{"title": "Paris", "text": "Paris is the capital of France."}],
    }
    rows = [
        {**base, "condition": "real", "verdict": "SUPPORTS"},
        {**base, "condition": "shuffled", "verdict": "REFUTES"},
        {**base, "condition": "real", "verdict": "NOT_ENOUGH_INFO"},
        {
            **base,
            "dataset": "other-dataset",
            "index": 2,
            "condition": "real",
            "verdict": "SUPPORTS",
        },
    ]
    output = build_references(
        rows,
        evidence_per_claim=1,
        include_nei=False,
        include_verifier_relation=True,
    )
    assert len(output) == 2
    assert len(output[0]["passages"]) == 1
    assert "Candidate relation: SUPPORTS" in output[0]["passages"][0]["text"]
    assert output[0]["shuffled_donor_dataset"] != output[0]["dataset"]


def test_build_references_hides_noisy_relation_by_default() -> None:
    rows = [
        {
            "dataset": dataset,
            "index": index,
            "claim_index": 0,
            "proposition": "A claim.",
            "condition": "real",
            "verdict": "SUPPORTS",
            "confidence": 0.99,
            "evidence": [{"title": "A", "text": "A candidate sentence."}],
        }
        for dataset, index in (("a", 1), ("b", 2))
    ]
    output = build_references(rows, evidence_per_claim=1, include_nei=False)
    assert "Candidate relation" not in output[0]["passages"][0]["text"]


def test_build_references_fills_uncovered_universe_rows() -> None:
    rows = [
        {
            "dataset": "a",
            "index": 1,
            "claim_index": 0,
            "proposition": "A claim.",
            "condition": "real",
            "verdict": "SUPPORTS",
            "confidence": 0.8,
            "evidence": [{"title": "A", "text": "A relevant sentence."}],
        },
        {
            "dataset": "b",
            "index": 2,
            "claim_index": 0,
            "proposition": "Another claim.",
            "condition": "real",
            "verdict": "SUPPORTS",
            "confidence": 0.8,
            "evidence": [{"title": "B", "text": "Another relevant sentence."}],
        },
    ]
    universe = [
        {"dataset": "a", "index": 1},
        {"dataset": "a", "index": 3},
        {"dataset": "b", "index": 2},
    ]
    output = build_references(
        rows, evidence_per_claim=1, include_nei=False, universe=universe
    )
    uncovered = next(row for row in output if row["index"] == 3)
    assert uncovered["passages"] == []
    assert uncovered["shuffled_passages"] == []


def test_reference_budgets_bound_pathological_wikipedia_sentences() -> None:
    assert truncate_chars("abcdefghij", 8) == "abcd ..."
    passages = [
        {"title": "A", "text": "short"},
        {"title": "B", "text": "x" * 100},
        {"title": "C", "text": "also short"},
    ]
    limited = limit_reference_passages(passages, max_chars=40)
    assert [passage["title"] for passage in limited] == ["A", "C"]
    rendered = "\n".join(
        f"- {passage['title']}: {passage['text']}" for passage in limited
    )
    assert len(rendered) <= 40
