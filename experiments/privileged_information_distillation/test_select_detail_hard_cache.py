from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.select_detail_hard_cache import (
    select_records,
)


def test_select_records_keeps_detail_positives_and_matched_controls() -> None:
    records = [
        {
            "dataset": "varied",
            "index": 1,
            "label": 1,
            "parse_error": False,
            "label_match": True,
            "reasoning_summary": (
                "The core answer is correct, but it adds a false historical date."
            ),
        },
        {
            "dataset": "varied",
            "index": 2,
            "label": 1,
            "parse_error": False,
            "label_match": True,
            "reasoning_summary": "The central answer itself is false.",
        },
        {
            "dataset": "varied",
            "index": 3,
            "label": 0,
            "parse_error": False,
            "label_match": True,
            "reasoning_summary": "The answer is accurate.",
        },
        {
            "dataset": "varied",
            "index": 4,
            "label": 0,
            "parse_error": False,
            "label_match": True,
            "reasoning_summary": "The answer is accurate.",
        },
    ]

    selected, positives, negatives = select_records(
        records,
        seed=7,
        negative_ratio=1.0,
    )

    assert positives == 1
    assert negatives == 1
    assert {record["label"] for record in selected} == {0, 1}
    assert any(record["index"] == 1 for record in selected)


def test_select_records_is_stable_under_input_reordering() -> None:
    records = [
        {
            "dataset": "varied",
            "index": index,
            "label": label,
            "parse_error": False,
            "label_match": True,
            "reasoning_summary": summary,
        }
        for index, label, summary in [
            (1, 1, "The main answer is correct, but it adds a false detail."),
            (2, 0, "Accurate."),
            (3, 0, "Accurate."),
        ]
    ]

    first, _, _ = select_records(records, seed=2, negative_ratio=1.0)
    second, _, _ = select_records(list(reversed(records)), seed=2, negative_ratio=1.0)

    assert [(row["dataset"], row["index"]) for row in first] == [
        (row["dataset"], row["index"]) for row in second
    ]
