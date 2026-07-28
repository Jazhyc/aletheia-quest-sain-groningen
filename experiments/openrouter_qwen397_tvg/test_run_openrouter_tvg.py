from __future__ import annotations

import math
from argparse import Namespace

import pytest

from experiments.openrouter_qwen397_tvg.run_openrouter_tvg import (
    EXPLICIT_DIGITS_PROMPT_TEMPLATE,
    FROZEN_PROMPT_TEMPLATE,
    OpenRouterError,
    binary_score_from_top_logprobs,
    extract_terminal_binary_top_logprobs,
    prompt_template_for_variant,
    score_item,
    WorkItem,
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


def test_score_item_exports_binary_soft_cache_fields(monkeypatch) -> None:
    class Response:
        status_code = 200
        headers: dict[str, str] = {}

        def json(self):
            return {
                "id": "response",
                "model": "qwen/qwen3.5-397b-a17b",
                "provider": "Alibaba",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Prediction:1"},
                        "logprobs": {
                            "content": [
                                {"token": "Prediction", "top_logprobs": []},
                                {"token": ":", "top_logprobs": []},
                                {
                                    "token": "1",
                                    "top_logprobs": [
                                        {"token": "1", "logprob": math.log(0.8)},
                                        {"token": "0", "logprob": math.log(0.2)},
                                    ],
                                },
                            ]
                        },
                    }
                ],
            }

    monkeypatch.setattr(
        "experiments.openrouter_qwen397_tvg.run_openrouter_tvg.requests.post",
        lambda *args, **kwargs: Response(),
    )
    item = WorkItem("dataset", 1, 0, "prompt", "hash", 6)
    args = Namespace(
        model="qwen/qwen3.5-397b-a17b",
        max_tokens=8,
        provider_sort="price",
        allow_fallbacks=False,
        provider_only="Alibaba",
        top_logprobs=5,
        endpoint="https://example.test",
        request_timeout=1.0,
        max_retries=0,
    )

    record = score_item(item, args, "secret")

    assert record["parse_error"] is False
    assert record["missing_rating_token_ids"] == []
    assert record["target_logprobs"] == pytest.approx(
        {"honest": math.log(0.2), "deceptive": math.log(0.8)}
    )
    assert record["target_probs"] == pytest.approx(
        {"honest": 0.2, "deceptive": 0.8}
    )
