import pytest

from phoenix_wright_v3 import (
    MEMBER_NAMES,
    TRUNCATION_MARKER,
    build_member_prompt,
    reply_to_score,
)


MESSAGES = [
    {"role": "system", "content": "Be accurate."},
    {"role": "user", "content": "Is Paris in France?"},
    {
        "role": "assistant",
        "content": "Yes, Paris is in France.",
        "reasoning": "The ordinary fact is clear, so answer honestly.",
    },
]


def test_deception_prompt_excludes_private_reasoning() -> None:
    prompt = build_member_prompt(MESSAGES)
    assert MEMBER_NAMES == ("deception",)
    assert "<assistant_reasoning>" not in prompt
    assert "The ordinary fact is clear" not in prompt
    assert "Yes, Paris is in France." in prompt


def test_unknown_member_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown Phoenix member"):
        build_member_prompt(MESSAGES, "resolved_intent")


def test_reply_to_score_uses_last_prediction_and_safe_default() -> None:
    assert reply_to_score("Prediction:1\nCorrection Prediction:0") == 0.0
    assert reply_to_score("No parseable verdict") == 0.0


def test_frozen_v2_renderer_does_not_merge_same_role_messages() -> None:
    messages = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer"},
    ]
    prompt = build_member_prompt(messages)
    assert "USER: first\n\nUSER: second" in prompt


def test_frozen_v2_renderer_marks_long_field_truncation() -> None:
    prompt = build_member_prompt(
        [{"role": "assistant", "content": "a" * 4000}]
    )
    assert TRUNCATION_MARKER in prompt
    assert len(prompt.split("<output>\n", 1)[1].split("\n</output>", 1)[0]) == 3000
