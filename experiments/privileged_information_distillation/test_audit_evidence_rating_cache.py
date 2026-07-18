from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.audit_evidence_rating_cache import (
    audit_records,
)


def make_record(index: int, label: int, rating: int) -> dict[str, object]:
    return {
        "dataset": "varied",
        "index": index,
        "label": label,
        "rating": rating,
        "parse_error": False,
        "label_match": True,
        "rating_polarity_match": True,
        "reasoning_summary": "Concrete visible evidence.",
        "student_target": f"Rating:{rating}\nPrediction:{label}",
    }


def test_balanced_diverse_cache_passes() -> None:
    records = [
        make_record(0, 0, 1),
        make_record(1, 0, 2),
        make_record(2, 0, 3),
        make_record(3, 1, 5),
        make_record(4, 1, 6),
        make_record(5, 1, 7),
    ]
    report = audit_records(
        records,
        expected_total=6,
        minimum_usable=6,
        minimum_distinct_ratings=4,
        minimum_intermediate_fraction=0.25,
    )
    assert report["passed"] is True
    assert report["intermediate_count"] == 4


def test_extreme_only_cache_fails_diversity_gates() -> None:
    records = [make_record(0, 0, 1), make_record(1, 1, 7)]
    report = audit_records(
        records,
        expected_total=2,
        minimum_usable=2,
        minimum_distinct_ratings=4,
        minimum_intermediate_fraction=0.25,
    )
    assert report["passed"] is False
    assert report["distinct_ratings"] == 2
    assert report["intermediate_fraction"] == 0.0


def test_privileged_leak_is_excluded_and_fails_gate() -> None:
    record = make_record(0, 0, 2)
    record["reasoning_summary"] = "The privileged label says this is honest."
    report = audit_records(
        [record],
        expected_total=1,
        minimum_usable=1,
        minimum_distinct_ratings=1,
        minimum_intermediate_fraction=0.0,
    )
    assert report["passed"] is False
    assert report["usable"] == 0
    assert report["leak_counts"] == {"privileged_information": 1}
