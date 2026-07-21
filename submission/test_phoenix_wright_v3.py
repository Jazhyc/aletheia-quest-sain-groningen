import joblib
import numpy as np

from phoenix_wright_v3 import (
    MEMBER_NAMES,
    build_member_prompt,
    has_reasoning,
    ngram_text_view,
    reply_to_score,
    stack_scores,
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


def test_member_prompts_route_reasoning_only_to_intent() -> None:
    assert "<assistant_reasoning>" not in build_member_prompt(MESSAGES, "deception")
    assert "<assistant_reasoning>" in build_member_prompt(MESSAGES, "resolved_intent")
    assert has_reasoning(MESSAGES)


def test_ngram_view_has_frozen_delimiters() -> None:
    text = ngram_text_view(MESSAGES)
    assert text.startswith("CONTEXT:\nSYSTEM: Be accurate.")
    assert text.endswith("FINAL_ASSISTANT:\nYes, Paris is in France.")
    assert "reasoning" not in text.lower()


def test_reply_to_score_uses_last_prediction_and_safe_default() -> None:
    assert reply_to_score("Prediction:1\nCorrection Prediction:0") == 0.0
    assert reply_to_score("No parseable verdict") == 0.0


def test_packaged_stack_returns_continuous_scores() -> None:
    artifact = joblib.load("submission/phoenix_wright_v3_ngram.joblib")
    members = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    scores = stack_scores([MESSAGES, MESSAGES], members, artifact)
    assert tuple(artifact["member_names"]) == MEMBER_NAMES
    assert scores.shape == (2,)
    assert 0.0 < scores[0] < scores[1] < 1.0
