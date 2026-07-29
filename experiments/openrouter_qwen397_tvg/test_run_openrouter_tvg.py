from __future__ import annotations

import math

import pytest

from experiments.openrouter_qwen397_tvg.run_openrouter_tvg import (
    OpenRouterError,
    binary_score_from_top_logprobs,
    extract_terminal_binary_top_logprobs,
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
