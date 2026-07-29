from __future__ import annotations

import pytest

from experiments.q397_reasoning_router.run_prompt_sweep_ndif import (
    reply_audit,
    split_replies,
)


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        assert not add_special_tokens
        return text.split()


def test_split_replies_preserves_prompt_major_order() -> None:
    replies = [f"reply-{index}" for index in range(6)]

    split = split_replies(replies, rows=2)

    assert split == {
        "summary_baseline": ["reply-0", "reply-1"],
        "claim_check": ["reply-2", "reply-3"],
        "balanced_audit": ["reply-4", "reply-5"],
    }


def test_split_replies_rejects_incomplete_results() -> None:
    with pytest.raises(ValueError, match="expected 6 replies, found 5"):
        split_replies(["reply"] * 5, rows=2)


def test_reply_audit_counts_missing_prediction_boundaries() -> None:
    audit = reply_audit(
        _WhitespaceTokenizer(),
        [
            "short Prediction: 0",
            "one two three four",
            "longer response Prediction: 1",
        ],
    )

    assert audit["fallback_boundaries"] == 1
    assert audit["reply_tokens"] == {
        "minimum": 3,
        "median": 4.0,
        "p95": 4.0,
        "maximum": 4,
    }
