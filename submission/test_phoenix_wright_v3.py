import joblib
import numpy as np

from phoenix_wright_v3 import (
    MEMBER_NAMES,
    build_member_prompt,
    has_reasoning,
    ngram_text_view,
    plan_intent_queries,
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


def test_intent_gate_preserves_endpoint_decisions() -> None:
    artifact = joblib.load("submission/phoenix_wright_v3_ngram.joblib")
    no_reasoning = [dict(message) for message in MESSAGES]
    no_reasoning[-1].pop("reasoning")
    messages = [MESSAGES, no_reasoning]
    deception = np.asarray([0.0, 0.0])
    query_mask, intent_features = plan_intent_queries(
        messages, deception, artifact, threshold=0.5
    )

    assert query_mask.shape == (2,)
    assert not query_mask[1]
    assert intent_features.tolist() == [0.5, 0.0]

    score_if_zero = stack_scores(
        messages, np.column_stack([deception, np.zeros(2)]), artifact
    )
    score_if_one = stack_scores(
        messages, np.column_stack([deception, np.ones(2)]), artifact
    )
    for position in np.flatnonzero(~query_mask):
        if has_reasoning(messages[position]):
            assert (score_if_zero[position] >= 0.5) == (
                score_if_one[position] >= 0.5
            )
