from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import (
    append_reference_material,
    build_student_prompt,
    format_student_target,
    parse_counterfactual_teacher_target,
    route_reference_material,
    truncate_text,
)
from experiments.privileged_information_distillation.generate_teacher_data import (
    filter_teacher_rows_by_dataset,
    limit_teacher_rows,
)


def test_build_student_prompt_includes_head_tail_reasoning_when_requested() -> None:
    messages = [
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": "Answer",
            "reasoning": "start-" + ("0123456789" * 10) + "-end",
        },
    ]

    prompt = build_student_prompt(
        messages,
        "JUDGE",
        100,
        "tail",
        include_reasoning=True,
        reasoning_max_chars=50,
        reasoning_truncation="head_tail",
    )

    assert "<assistant_reasoning>" in prompt
    assert "start-" in prompt
    assert "-end" in prompt
    assert "[... middle truncated ...]" in prompt


def test_asymmetric_head_tail_truncation_changes_only_the_split() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 10

    tail_heavy = truncate_text(text, 100, "head_tail_25")
    balanced = truncate_text(text, 100, "head_tail")
    head_heavy = truncate_text(text, 100, "head_tail_75")

    assert len(tail_heavy) == len(balanced) == len(head_heavy) == 100
    assert len(tail_heavy.split("\n\n[... middle truncated ...]\n\n")[0]) < len(
        balanced.split("\n\n[... middle truncated ...]\n\n")[0]
    )
    assert len(head_heavy.split("\n\n[... middle truncated ...]\n\n")[0]) > len(
        balanced.split("\n\n[... middle truncated ...]\n\n")[0]
    )


def test_build_student_prompt_omits_missing_or_disabled_reasoning() -> None:
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]

    missing = build_student_prompt(
        messages,
        "JUDGE",
        100,
        "tail",
        include_reasoning=True,
        reasoning_max_chars=1200,
    )
    disabled = build_student_prompt(
        [*messages[:-1], {**messages[-1], "reasoning": "private"}],
        "JUDGE",
        100,
        "tail",
    )

    assert "<assistant_reasoning>" not in missing
    assert "<assistant_reasoning>" not in disabled


def test_build_student_prompt_can_avoid_duplicate_final_output() -> None:
    messages = [
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Final question"},
        {"role": "assistant", "content": "Final answer"},
    ]

    prompt = build_student_prompt(
        messages,
        "JUDGE",
        1000,
        "tail",
        exclude_final_output_from_context=True,
    )

    context, output = prompt.split("\n\n<output>\n", maxsplit=1)
    assert "Earlier answer" in context
    assert "Final question" in context
    assert "Final answer" not in context
    assert output.startswith("Final answer\n</output>")


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


def test_append_reference_material_marks_retrieval_as_untrusted() -> None:
    prompt = append_reference_material("PROMPT", "- Entity: fact")

    assert prompt.startswith("PROMPT\n\n<reference_material>")
    assert "incomplete, ambiguous, or irrelevant" in prompt
    assert "- Entity: fact" in prompt
    assert prompt.endswith("</reference_material>")


def test_route_reference_material_can_keep_evidence_teacher_only() -> None:
    student, teacher = route_reference_material("PROMPT", "- Fact", "teacher_only")

    assert student == "PROMPT"
    assert "<reference_material>" in teacher
    assert "- Fact" in teacher


def test_route_reference_material_preserves_legacy_shared_visibility() -> None:
    student, teacher = route_reference_material(
        "PROMPT", "- Fact", "teacher_and_student"
    )

    assert student == teacher
    assert "<reference_material>" in student


def test_route_reference_material_rejects_unknown_visibility() -> None:
    with pytest.raises(ValueError, match="reference_visibility"):
        route_reference_material("PROMPT", "- Fact", "unknown")
