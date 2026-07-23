from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.continuous_hybrid.run_qwen_organism_margin import (
    false_score_from_logprobs,
    single_token_id,
)
from experiments.continuous_hybrid.evaluate_locked_test import (
    decoupled_macro_metrics,
    fit_locked_stack,
)
from experiments.continuous_hybrid.stack_cached_features import (
    grouped_oof_scores,
    organism_margin_acceptance,
)


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


def test_organism_margin_acceptance_requires_both_groupings() -> None:
    def report(ba: float, auroc: float, varied: float, qwen: float) -> dict:
        return {
            "overall": {"balanced_accuracy": ba, "auroc": auroc},
            "scenarios": {"varied-deception": {"metrics": {"balanced_accuracy": varied}}},
            "families": {"Qwen": {"metrics": {"balanced_accuracy": qwen}}},
        }

    evaluations = {
        "compact_continuous": {
            "groupings": {
                "dataset": report(0.93, 0.95, 0.84, 0.91),
                "organism": report(0.93, 0.95, 0.84, 0.91),
            }
        },
        "compact_plus_continuous_organism": {
            "groupings": {
                "dataset": report(0.933, 0.96, 0.84, 0.912),
                "organism": report(0.931, 0.96, 0.84, 0.912),
            }
        },
    }
    acceptance = organism_margin_acceptance(evaluations)
    assert acceptance is not None
    assert acceptance["accepted"] is False
    assert acceptance["comparisons"]["dataset"]["passed"] is True
    assert acceptance["comparisons"]["organism"]["passed"] is False


def test_locked_stack_uses_fixed_half_probability_boundary() -> None:
    validation = pd.DataFrame({
        "x": [-2.0, -1.0, 1.0, 2.0],
        "label": [0, 0, 1, 1],
    })
    test = pd.DataFrame({"x": [-1.5, 1.5]})
    _, scores = fit_locked_stack(validation, test, ["x"])
    assert (scores.to_numpy() >= 0.5).tolist() == [False, True]


def test_decoupled_metrics_use_boolean_and_score_independently() -> None:
    frame = pd.DataFrame({
        "dataset": ["a"] * 4,
        "label": [0, 0, 1, 1],
        "deceptive": [False, False, True, True],
        "score": [0.9, 0.8, 0.2, 0.1],
    })
    metrics = decoupled_macro_metrics(frame)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["auroc"] == 0.0
