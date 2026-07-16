from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.continuous_hybrid.run_qwen_organism_margin import (
    false_score_from_logprobs,
    single_token_id,
)
from experiments.continuous_hybrid.stack_cached_features import grouped_oof_scores


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return {"TRUE": [10], "FALSE": [20], "two tokens": [1, 2]}[text]


def test_single_token_id_rejects_multi_token_verdict() -> None:
    assert single_token_id(FakeTokenizer(), "TRUE") == 10
    try:
        single_token_id(FakeTokenizer(), "two tokens")
    except ValueError as error:
        assert "not a single token" in str(error)
    else:
        raise AssertionError("multi-token verdict was accepted")


def test_false_score_normalizes_constrained_logprobs() -> None:
    score = false_score_from_logprobs({10: -2.0, 20: -1.0}, 10, 20)
    assert score is not None
    assert np.isclose(score, 1.0 / (1.0 + np.exp(-1.0)))
    assert false_score_from_logprobs({10: -2.0}, 10, 20) is None


def test_grouped_oof_scores_assign_every_held_out_group() -> None:
    rows = []
    for group in ("a", "b", "c"):
        for label in (0, 1):
            for repeat in range(2):
                rows.append({
                    "dataset": group,
                    "label": label,
                    "feature": float(label) + repeat * 0.01,
                })
    frame = pd.DataFrame(rows)
    scores = grouped_oof_scores(frame, ["feature"], "dataset")
    assert scores.shape == (len(frame),)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.5).astype(int) == frame["label"].to_numpy()).all()
