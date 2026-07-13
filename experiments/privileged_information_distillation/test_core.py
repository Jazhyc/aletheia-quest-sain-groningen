from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import (
    format_student_target,
    parse_counterfactual_teacher_target,
)
from experiments.privileged_information_distillation.generate_teacher_data import (
    filter_teacher_rows_by_dataset,
    limit_teacher_rows,
)


def test_parse_counterfactual_target_with_privileged_prediction_fallback() -> None:
    raw = """assistantanalysisHidden reasoning.
assistantfinal<reasoning_summary>Short summary.</reasoning_summary>
<facts>The correct fact.</facts>
<contradiction>The output conflicts with the fact.</contradiction>"""

    assert parse_counterfactual_teacher_target(raw, expected_prediction=1) == (
        "Short summary.",
        "The correct fact.",
        "The output conflicts with the fact.",
        1,
    )


def test_format_counterfactual_target_preserves_required_order() -> None:
    target = format_student_target("Summary", 0, "Facts", "No contradiction")

    assert target == (
        "<reasoning_summary>\nSummary\n</reasoning_summary>\n"
        "<facts>\nFacts\n</facts>\n"
        "<contradiction>\nNo contradiction\n</contradiction>\n"
        "Prediction:0"
    )


def test_limit_teacher_rows_selects_balanced_labels_from_ordered_rows() -> None:
    rows = [{"label": 0, "index": i} for i in range(5)] + [
        {"label": 1, "index": i} for i in range(5, 10)
    ]

    selected = limit_teacher_rows(rows, limit=None, limit_per_label=2)

    assert [row["label"] for row in selected] == [0, 0, 1, 1]


def test_filter_teacher_rows_by_dataset() -> None:
    rows = [
        {"dataset": "dev-instructed-deception-model", "index": 1},
        {"dataset": "dev-varied-deception-model", "index": 2},
    ]

    selected = filter_teacher_rows_by_dataset(rows, "varied-deception")

    assert selected == [rows[1]]


def test_filter_teacher_rows_by_dataset_rejects_empty_selection() -> None:
    rows = [{"dataset": "dev-instructed-deception-model", "index": 1}]

    with pytest.raises(RuntimeError, match="no teacher rows match"):
        filter_teacher_rows_by_dataset(rows, "varied-deception")
