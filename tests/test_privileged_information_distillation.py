import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import (
    build_student_prompt,
    build_teacher_prompt,
    extract_harmony_final,
    format_student_target,
    parse_teacher_target,
)
from experiments.privileged_information_distillation.generate_teacher_data import (
    cache_matches,
    reparse_cached_record,
)


def test_teacher_prompt_contains_label_but_student_prompt_does_not() -> None:
    messages = [
        {"role": "user", "content": "Is the moon made of cheese?"},
        {"role": "assistant", "content": "Yes."},
    ]
    student = build_student_prompt(messages, "Judge the output.", 3000, "tail")
    teacher = build_teacher_prompt(
        student,
        "Authoritative label: __GROUND_TRUTH__. Prediction:__GROUND_TRUTH__",
        1,
    )

    assert "GROUND_TRUTH" not in teacher
    assert "Authoritative label: 1" in teacher
    assert "Prediction:1" in teacher
    assert "Authoritative label" not in student
    assert "<context>" in teacher and "<output>" in teacher


def test_parser_discards_harmony_analysis_and_keeps_summary() -> None:
    raw = (
        "analysisThis private trace is verbose and says Prediction:0."
        "assistantfinal<reasoning_summary>\n"
        "The answer reverses the known fact and would mislead the user.\n"
        "</reasoning_summary>\nPrediction:1"
    )

    assert extract_harmony_final(raw).startswith("<reasoning_summary>")
    assert parse_teacher_target(raw) == (
        "The answer reverses the known fact and would mislead the user.",
        1,
    )


def test_parser_requires_structured_final_target() -> None:
    raw = "analysisI think the label is 1.assistantfinalPrediction:1"

    assert parse_teacher_target(raw) is None


def test_parser_can_attach_authoritative_prediction_to_summary_only() -> None:
    raw = (
        "analysisPrivate work.assistantfinal"
        "<reasoning_summary>The output contradicts the known fact.</reasoning_summary>"
    )

    assert parse_teacher_target(raw, expected_prediction=1) == (
        "The output contradicts the known fact.",
        1,
    )


def test_student_target_has_strict_binary_contract() -> None:
    assert format_student_target("Concrete factual contrast.", 0) == (
        "<reasoning_summary>\nConcrete factual contrast.\n</reasoning_summary>\n"
        "Prediction:0"
    )


def test_cache_requires_exact_prompts_and_valid_expected_prediction() -> None:
    row = {
        "dataset": "dataset",
        "index": 3,
        "label": 1,
        "student_prompt": "student-v2",
        "teacher_prompt": "teacher-v2",
    }
    cached = {
        **row,
        "parse_error": False,
        "label_match": True,
        "student_target": "<reasoning_summary>x</reasoning_summary>Prediction:1",
    }

    assert cache_matches(row, cached)
    assert not cache_matches(row, {**cached, "teacher_prompt": "teacher-v1"})
    assert not cache_matches(row, {**cached, "label_match": False})


def test_cached_summary_is_reparsed_without_harmony_analysis() -> None:
    row = {
        "dataset": "dataset",
        "index": 3,
        "label": 0,
        "student_prompt": "student",
        "teacher_prompt": "teacher",
    }
    cached = {
        **row,
        "raw_completion": (
            "analysisIgnore this.assistantfinal"
            "<reasoning_summary>The answer states the correct fact.</reasoning_summary>"
        ),
        "harmony_final": (
            "<reasoning_summary>The answer states the correct fact.</reasoning_summary>"
        ),
        "parse_error": True,
        "label_match": False,
    }

    refreshed = reparse_cached_record(row, cached)

    assert refreshed is not None
    assert refreshed["prediction_source"] == "privileged_label_fallback"
    assert refreshed["student_target"].endswith("Prediction:0")
    assert cache_matches(row, refreshed)
