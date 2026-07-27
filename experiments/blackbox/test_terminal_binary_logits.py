from __future__ import annotations

import pytest

from experiments.blackbox.run_judge import terminal_binary_target_probs


TARGETS = [
    {"name": "honest", "score": 0.0, "ids": [10]},
    {"name": "deceptive", "score": 1.0, "ids": [11]},
]


def test_terminal_binary_target_probs_uses_final_prediction_position() -> None:
    result = terminal_binary_target_probs(
        "Reasoning mentions 0.\nPrediction:1",
        [10, 99, 11],
        [
            {10: -0.1, 11: -2.0},
            {},
            {10: -2.0, 11: -0.1},
        ],
        targets=TARGETS,
        missing_logprob=-30.0,
    )

    assert result is not None
    prediction, probabilities = result
    assert prediction == 1
    assert probabilities["deceptive"] > probabilities["honest"]


def test_terminal_binary_target_probs_rejects_mismatched_sampled_label() -> None:
    result = terminal_binary_target_probs(
        "Reasoning.\nPrediction:1",
        [10],
        [{10: -0.1, 11: -2.0}],
        targets=TARGETS,
        missing_logprob=-30.0,
    )

    assert result is None


def test_terminal_binary_target_probs_rejects_missing_terminal_contract() -> None:
    result = terminal_binary_target_probs(
        "The answer is probably deceptive.",
        [11],
        [{10: -2.0, 11: -0.1}],
        targets=TARGETS,
        missing_logprob=-30.0,
    )

    assert result is None
