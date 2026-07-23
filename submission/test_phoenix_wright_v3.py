import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from phoenix_wright_v3 import (
    MEMBER_NAMES,
    REASONING_TRUNCATION_MARKER,
    TRUNCATION_MARKER,
    build_member_prompt,
    final_assistant_reasoning,
    reply_to_score,
    stack_reasoning_scores,
    truncate_reasoning_head_tail,
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


def test_reasoning_stack_uses_only_final_assistant_reasoning() -> None:
    messages = [
        {"role": "assistant", "content": "draft", "reasoning": "old plan"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "answer", "reasoning": "final plan"},
    ]

    assert final_assistant_reasoning(messages) == "final plan"


def test_reasoning_truncation_matches_training_contract() -> None:
    text = "a" * 3000 + "b" * 3000
    truncated = truncate_reasoning_head_tail(text)

    assert len(truncated) == 4000
    assert REASONING_TRUNCATION_MARKER in truncated
    assert truncated.startswith("a")
    assert truncated.endswith("b")


class _FakeVectorizer:
    def transform(self, text):
        return np.asarray([[float("deceive" in value)] for value in text])


class _FakeClassifier:
    def predict_proba(self, features):
        positive = 0.8 * np.asarray(features)[:, 0] + 0.1
        return np.column_stack([1.0 - positive, positive])


def test_reasoning_stack_falls_back_to_phoenix_when_trace_is_absent() -> None:
    artifact = {
        "format_version": 1,
        "reasoning_max_chars": 4000,
        "meta_feature_order": ["phoenix_score", "reasoning_ngram_score"],
        "vectorizer": _FakeVectorizer(),
        "ngram_classifier": _FakeClassifier(),
        "meta_coefficient": [2.0, 4.0],
        "meta_intercept": -2.0,
    }
    messages = [
        [{"role": "assistant", "content": "answer"}],
        [{
            "role": "assistant",
            "content": "answer",
            "reasoning": "I will deceive",
        }],
    ]

    scores = stack_reasoning_scores(messages, np.asarray([1.0, 0.0]), artifact)

    assert scores[0] == 1.0
    assert scores[1] > 0.5


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


def test_submission_bundles_rank24_phoenix_3_1_adapter() -> None:
    adapter_dir = Path(__file__).parent / "phoenix_wright_v3_adapters" / "deception"
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    weights = adapter_dir / "adapter_model.safetensors"

    assert config["base_model_name_or_path"] == "Qwen/Qwen3.5-9B"
    assert config["r"] == 24
    assert config["lora_alpha"] == 48
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == (
        "714e9a71f63780158ed8a29a9c408ad94c8e44396b37be7178212bf3e5c6303c"
    )


def test_submission_bundles_frozen_reasoning_stack() -> None:
    artifact_path = (
        Path(__file__).parent / "phoenix_wright_v3_reasoning_stack.joblib"
    )
    artifact = joblib.load(artifact_path)

    assert artifact["method"] == "phoenix_wright_v3.2_reasoning_ngram_logistic"
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == (
        "3db505e0d5e5e2ec333c1f9ca339479746e7e79960d18a6ba78161849f54fe90"
    )
    assert artifact["reasoning_max_chars"] == 4000
    assert len(artifact["vectorizer"].vocabulary_) == 12_936
    assert artifact["meta_feature_order"] == [
        "phoenix_score",
        "reasoning_ngram_score",
    ]
    assert np.allclose(
        artifact["meta_coefficient"],
        [2.138585302289483, 3.8294402187139944],
    )
    assert artifact["meta_intercept"] == pytest.approx(-2.5039903572703337)
