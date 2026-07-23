import json

import pytest

from experiments.privileged_information_distillation.generate_teacher_data import (
    build_configured_teacher_prompt,
    cache_matches,
    select_teacher_rows_by_manifest,
    teacher_expected_prediction,
)


def test_blind_teacher_prompt_cannot_contain_ground_truth_placeholder() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        build_configured_teacher_prompt(
            "judge\n\n<context>evidence</context>",
            "The label is __GROUND_TRUTH__",
            1,
            uses_ground_truth=False,
        )


def test_blind_teacher_prompt_is_independent_of_label_value() -> None:
    student_prompt = "judge\n\n<context>evidence</context>"
    template = "Make your own decision."

    prompts = {
        build_configured_teacher_prompt(
            student_prompt,
            template,
            label,
            uses_ground_truth=False,
        )
        for label in (0, 1)
    }

    assert prompts == {"Make your own decision.\n\n<context>evidence</context>"}


def test_blind_teacher_has_no_expected_prediction_fallback() -> None:
    assert teacher_expected_prediction({
        "label": 1, "teacher_uses_ground_truth": False
    }) is None
    assert teacher_expected_prediction({
        "label": 1, "teacher_uses_ground_truth": True
    }) == 1


def test_blind_cache_reuses_parsed_label_mismatch() -> None:
    row = {
        "label": 1,
        "teacher_model": "teacher",
        "teacher_output_format": "harmony",
        "reasoning_effort": "medium",
        "teacher_uses_ground_truth": False,
        "student_prompt": "student",
        "teacher_prompt": "teacher prompt",
    }
    cached = {
        **row,
        "parse_error": False,
        "label_match": False,
        "student_target": "<reasoning_summary>Error.</reasoning_summary>Prediction:0",
    }

    assert cache_matches(row, cached)


def test_teacher_manifest_selection_validates_exact_rows(tmp_path) -> None:
    rows = [
        {"dataset": "dataset", "index": index, "label": index % 2}
        for index in range(3)
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(
        json.dumps(rows[index]) + "\n" for index in (0, 2)
    ))

    assert select_teacher_rows_by_manifest(rows, manifest) == [rows[0], rows[2]]
