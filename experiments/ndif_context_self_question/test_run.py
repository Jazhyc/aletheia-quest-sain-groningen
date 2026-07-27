from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.ndif_context_self_question.run import (
    PREDICTION_PREFIX,
    SELF_QUESTION,
    normalized_messages,
    render_self_question_prompt,
    validate_cached_group,
)


class FakeTokenizer:
    def __init__(self, *, supports_thinking: bool = True) -> None:
        self.supports_thinking = supports_thinking
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        if not self.supports_thinking and "enable_thinking" in kwargs:
            raise TypeError("unsupported")
        self.calls.append((deepcopy(messages), dict(kwargs)))
        return "rendered:"


def test_normalized_messages_preserves_only_role_and_content() -> None:
    messages = [
        {"role": "system", "content": "system", "private": "drop"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "reasoning": "drop"},
    ]
    assert normalized_messages(messages) == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_render_appends_self_question_without_mutating_source() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    original = deepcopy(messages)
    prompt = render_self_question_prompt(tokenizer, messages)

    assert messages == original
    rendered, kwargs = tokenizer.calls[0]
    assert rendered[:-1] == original
    assert rendered[-1] == {"role": "user", "content": SELF_QUESTION}
    assert kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert prompt == "rendered:" + PREDICTION_PREFIX


def test_render_retries_without_enable_thinking() -> None:
    tokenizer = FakeTokenizer(supports_thinking=False)
    prompt = render_self_question_prompt(
        tokenizer,
        [{"role": "assistant", "content": "answer"}],
    )
    assert tokenizer.calls[0][1] == {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    assert prompt.endswith(PREDICTION_PREFIX)


def test_cache_validation_binds_prompt_and_exact_organism() -> None:
    cached = {
        "model": "model",
        "lora": "adapter",
        "keys": [["dataset", "1"]],
        "prompt_sha256": ["hash"],
        "scores": [0.25],
    }
    validate_cached_group(
        cached,
        model_id="model",
        lora_id="adapter",
        keys=[["dataset", "1"]],
        hashes=["hash"],
    )
    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_cached_group(
            cached,
            model_id="model",
            lora_id="adapter",
            keys=[["dataset", "1"]],
            hashes=["different"],
        )
