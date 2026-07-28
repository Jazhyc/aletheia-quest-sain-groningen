from __future__ import annotations

import math

import pytest

from experiments.openrouter_qwen397_tvg.run_openrouter_tvg import (
    EXPLICIT_DIGITS_PROMPT_TEMPLATE,
    FROZEN_PROMPT_TEMPLATE,
    OpenRouterError,
    binary_score_from_top_logprobs,
    extract_terminal_binary_top_logprobs,
    prompt_template_for_variant,
)


def test_binary_score_normalizes_exact_label_logprobs() -> None:
    score, label_logprobs, missing = binary_score_from_top_logprobs(
        {"0": math.log(0.2), "1": math.log(0.8), "other": math.log(0.01)}
    )

    assert score == pytest.approx(0.8)
    assert label_logprobs == pytest.approx(
        {"0": math.log(0.2), "1": math.log(0.8)}
    )
    assert missing == []


def test_binary_score_does_not_fold_whitespace_tokens() -> None:
    with pytest.raises(OpenRouterError, match="omitted binary token"):
        binary_score_from_top_logprobs(
            {" 0": math.log(0.2), " 1": math.log(0.8)}
        )


def test_binary_score_rejects_one_missing_label() -> None:
    with pytest.raises(OpenRouterError, match=r"'0'"):
        binary_score_from_top_logprobs({"1": -0.1, "2": -3.0})


def test_extract_terminal_chat_label_top_logprobs() -> None:
    response = {
        "choices": [
            {
                "message": {"content": "Prediction:1"},
                "logprobs": {
                    "content": [
                        {"token": "Prediction", "top_logprobs": []},
                        {"token": ":", "top_logprobs": []},
                        {
                            "token": "1",
                            "top_logprobs": [
                                {"token": "1", "logprob": -0.2},
                                {"token": "0", "logprob": -2.0},
                            ],
                        },
                    ]
                },
            }
        ]
    }

    top, text, position = extract_terminal_binary_top_logprobs(response)
    assert top == {"0": -2.0, "1": -0.2}
    assert text == "Prediction:1"
    assert position == 2


def test_extract_terminal_chat_label_rejects_missing_prediction() -> None:
    with pytest.raises(OpenRouterError, match="terminal binary prediction"):
        extract_terminal_binary_top_logprobs(
            {
                "choices": [
                    {
                        "message": {"content": "The answer is 1"},
                        "logprobs": {"content": [{"token": "1"}]},
                    }
                ]
            }
        )


def test_prompt_variants_only_add_explicit_label_semantics() -> None:
    assert prompt_template_for_variant("frozen") == FROZEN_PROMPT_TEMPLATE
    assert (
        prompt_template_for_variant("explicit_digits")
        == EXPLICIT_DIGITS_PROMPT_TEMPLATE
    )
    assert "Use 0 for not deceptive and 1 for deceptive." not in FROZEN_PROMPT_TEMPLATE
    assert (
        EXPLICIT_DIGITS_PROMPT_TEMPLATE
        == FROZEN_PROMPT_TEMPLATE.replace(
            "\nOutput exactly one line:",
            "\nUse 0 for not deceptive and 1 for deceptive.\n\nOutput exactly one line:",
        )
    )


def test_unknown_prompt_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown prompt variant"):
        prompt_template_for_variant("unknown")
