from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.selective_hop import (
    bounded_candidates,
    build_audit_prompt,
    parse_audit_completion,
)
from experiments.fever_fact_verification.evaluate_selective_audit import summarize
from experiments.fever_fact_verification.build_selective_reference import build_cache


def test_parse_decisive_candidate() -> None:
    candidates = [{"title": "Padstow", "text": "Padstow is in Cornwall."}]
    parsed = parse_audit_completion(
        """CANDIDATE_1: DECISIVE_CONTRADICTION
SELECTED_ID: 1
DECISION: DECISIVE_CONTRADICTION
QUERY: NONE
RATIONALE: Cornwall is incompatible with Devon.""",
        candidates=candidates,
        proposition="Padstow is in Devon.",
    )
    assert parsed["parse_valid"]
    assert parsed["selected_candidate"] == candidates[0]
    assert parsed["query"] == ""


def test_invalid_selection_falls_back_to_abstention_query() -> None:
    candidates = [{"title": "Padstow", "text": "Padstow is in Cornwall."}]
    parsed = parse_audit_completion(
        """CANDIDATE_1: TOPICAL
SELECTED_ID: 1
DECISION: DECISIVE_CONTRADICTION
QUERY: NONE
RATIONALE: mismatch""",
        candidates=candidates,
        proposition="Padstow is in Devon.",
    )
    assert not parsed["parse_valid"]
    assert parsed["decision"] == "ABSTAIN"
    assert parsed["query"] == "Padstow is in Devon."


def test_bounded_candidates_and_prompt_do_not_include_labels() -> None:
    row = {
        "label": 1,
        "teacher_assessment": "false",
        "evidence": [{"title": "T", "text": "x" * 100}],
    }
    candidates = bounded_candidates(
        row, source_field="evidence", limit=1, max_chars=20
    )
    prompt = build_audit_prompt("A proposition.", candidates)
    assert len(candidates[0]["text"]) == 20
    assert "teacher_assessment" not in prompt
    assert "ground-truth label" in prompt


def test_selective_summary_reports_abstention_and_precision() -> None:
    rows = [
        {
            "decision": "DECISIVE_SUPPORT",
            "teacher_assessment": "true",
            "parse_valid": True,
            "candidate_assessments": [{"relation": "DECISIVE_SUPPORT"}],
        },
        {
            "decision": "ABSTAIN",
            "teacher_assessment": "false",
            "parse_valid": True,
            "candidate_assessments": [{"relation": "TOPICAL"}],
        },
    ]
    report = summarize(rows)
    assert report["coverage"] == 0.5
    assert report["conditional_teacher_precision"] == 1.0
    assert report["decisions"]["ABSTAIN"] == 1


def test_selective_cache_uses_second_hop_only_after_initial_abstention() -> None:
    base = {
        "index": 1,
        "claim_index": 0,
        "proposition": "A claim.",
        "selected_candidate": {"title": "Initial", "text": "Initial evidence."},
        "stage": "initial",
    }
    initial = [
        {**base, "dataset": "a", "decision": "DECISIVE_SUPPORT"},
        {
            **base,
            "dataset": "b",
            "decision": "ABSTAIN",
            "selected_candidate": None,
        },
    ]
    second = [{
        **base,
        "dataset": "b",
        "decision": "DECISIVE_CONTRADICTION",
        "selected_candidate": {"title": "Second", "text": "Second evidence."},
        "stage": "second",
    }]
    output, summary = build_cache(
        initial,
        second,
        [{"dataset": "a", "index": 1}, {"dataset": "b", "index": 1}],
        max_reference_chars=5000,
    )
    assert summary["passages"] == 2
    assert summary["selected_stages"] == {"initial": 1, "second": 1}
    assert output[1]["passages"][0]["title"] == "Second"
