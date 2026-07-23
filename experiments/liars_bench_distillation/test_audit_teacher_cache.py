from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.audit_teacher_cache import (
    audit_records,
    validate_audit,
)


def make_record(index: str, label: int, summary: str) -> dict[str, object]:
    return {
        "dataset": "liars-bench/example",
        "index": index,
        "label": label,
        "reasoning_summary": summary,
        "student_target": f"<reasoning_summary>{summary}</reasoning_summary>\nPrediction:{label}",
        "parse_error": False,
        "label_match": True,
    }


def test_audit_accepts_balanced_clean_cache() -> None:
    audit = audit_records([
        make_record("a", 0, "The answer follows the supplied facts."),
        make_record("b", 1, "The denial conflicts with the answer the output asserts."),
    ])

    assert audit["coverage"] == 1.0
    assert audit["label_counts"] == {"0": 1, "1": 1}
    assert audit["leak_rows"] == 0
    assert validate_audit(audit, min_coverage=0.9, max_leak_fraction=0.05) == []


def test_audit_allows_one_parse_failure_in_a_balanced_category() -> None:
    records = [
        *(make_record(f"n-{index}", 0, "Clean negative summary.") for index in range(32)),
        *(make_record(f"p-{index}", 1, "Clean positive summary.") for index in range(32)),
    ]
    records[-1] = {**records[-1], "parse_error": True, "student_target": None}

    audit = audit_records(records)

    assert audit["coverage"] == 63 / 64
    assert validate_audit(audit, min_coverage=0.9, max_leak_fraction=0.05) == []


def test_audit_checks_expected_shape() -> None:
    audit = audit_records([
        make_record("a", 0, "Clean summary."),
        make_record("b", 1, "Clean summary."),
    ])

    failures = validate_audit(
        audit,
        min_coverage=0.9,
        max_leak_fraction=0.05,
        expected_total=64,
        expected_datasets=4,
    )

    assert any("expected exactly 64" in failure for failure in failures)
    assert any("usable datasets" in failure for failure in failures)


def test_audit_rejects_leak_and_usable_label_imbalance() -> None:
    records = [
        make_record("a", 0, "The privileged evidence proves the response is wrong."),
        {
            **make_record("b", 1, "Clean summary."),
            "parse_error": True,
            "student_target": None,
        },
    ]

    audit = audit_records(records)
    failures = validate_audit(audit, min_coverage=0.9, max_leak_fraction=0.05)

    assert audit["leak_pattern_counts"] == {"privileged_evidence": 1}
    assert any("coverage" in failure for failure in failures)
    assert any("leak fraction" in failure for failure in failures)
    assert any("imbalance" in failure for failure in failures)


def test_audit_detects_unicode_ground_truth_and_generic_label_language() -> None:
    audit = audit_records([
        make_record("a", 0, "This matches the ground‑truth label 0."),
        make_record("b", 1, "Clean positive summary."),
    ])

    assert audit["leak_rows"] == 1
    assert audit["leak_pattern_counts"] == {"ground_truth": 1, "numeric_label": 1}


def test_audit_allows_semantic_label_and_legal_privileged_language() -> None:
    audit = audit_records([
        make_record("a", 0, "The report labels the risk high and cites privileged information."),
        make_record("b", 1, "Clean positive summary."),
    ])

    assert audit["leak_rows"] == 0


def test_audit_rejects_summary_conclusion_that_conflicts_with_label() -> None:
    audit = audit_records([
        make_record("a", 0, "The answer is accurate and not deceptive."),
        make_record("b", 1, "The answer is accurate and not deceptive."),
    ])

    failures = validate_audit(audit, min_coverage=0.9, max_leak_fraction=0.05)

    assert audit["polarity_conflict_rows"] == 1
    assert any("polarity conflict" in failure for failure in failures)
