import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.qwen_grpo_lora.run_qwen_grpo_lora import (
    make_correctness_reward,
    make_format_reward,
    parse_rating,
    structured_completion_match,
)


GOOD_POSITIVE = "<reasoning>\nEvidence says deceptive.\n</reasoning>\nPrediction:1"
GOOD_NEGATIVE = "<reasoning>\nEvidence says safe.\n</reasoning>\nPrediction:0"
SPACED_POSITIVE = "<reasoning>\nEvidence says deceptive.\n</reasoning>\nPrediction: 1"


def test_strict_binary_completion_requires_configured_separator():
    assert (
        structured_completion_match(
            GOOD_POSITIVE,
            rating_min=0,
            rating_max=1,
            output_label="Prediction",
            output_separator="",
        )
        is not None
    )
    assert (
        structured_completion_match(
            SPACED_POSITIVE,
            rating_min=0,
            rating_max=1,
            output_label="Prediction",
            output_separator="",
        )
        is None
    )


def test_strict_parse_rejects_flexible_spacing_fallbacks():
    assert (
        parse_rating(
            GOOD_POSITIVE,
            rating_min=0,
            rating_max=1,
            output_label="Prediction",
            output_separator="",
            strict_structured=True,
        )
        == 1
    )
    assert (
        parse_rating(
            SPACED_POSITIVE,
            rating_min=0,
            rating_max=1,
            output_label="Prediction",
            output_separator="",
            strict_structured=True,
        )
        is None
    )


def test_flexible_parse_still_accepts_legacy_spacing():
    assert (
        parse_rating(
            SPACED_POSITIVE,
            rating_min=0,
            rating_max=1,
            output_label="Prediction",
            output_separator=None,
            strict_structured=False,
        )
        == 1
    )


def test_rewards_use_strict_binary_contract_for_future_runs():
    correctness = make_correctness_reward(
        rating_min=0,
        rating_max=1,
        output_label="Prediction",
        output_separator="",
        strict_output_format=True,
    )
    fmt = make_format_reward(
        rating_min=0,
        rating_max=1,
        output_label="Prediction",
        output_separator="",
    )

    completions = [GOOD_POSITIVE, GOOD_NEGATIVE, SPACED_POSITIVE]
    assert correctness(completions, label=[1, 0, 1]) == [1.0, 1.0, 0.0]
    assert fmt(completions) == [1.0, 1.0, 0.0]
