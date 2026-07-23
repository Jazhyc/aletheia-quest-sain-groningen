import hashlib
from pathlib import Path

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


def test_submission_bundles_exact_ndif_client_wheels() -> None:
    submission_dir = Path(__file__).parent
    expected = {
        "transformers-5.15.0.dev0-py3-none-any.whl":
            "0dbdde6331c4562d429467c41c9d2dac641db6a8bd36dab97771a7ebfb6db83b",
        "nnsight-0.7.1.dev41+gd901da3ed-cp312-cp312-linux_x86_64.whl":
            "831bdcd9ad90a8e5ed94348f36e011d8aaa0002d9e570c955169efcd60d0b6ae",
    }
    requirements = (submission_dir / "requirements.txt").read_text()

    for filename, expected_sha256 in expected.items():
        wheel = submission_dir / "wheels" / filename
        assert f"./submission/wheels/{filename}" in requirements
        assert hashlib.sha256(wheel.read_bytes()).hexdigest() == expected_sha256
