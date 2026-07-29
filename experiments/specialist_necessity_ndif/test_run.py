from __future__ import annotations

import numpy as np

from experiments.specialist_necessity_ndif.run import (
    REASONING_MAX_CHARS,
    add_reasoning,
    binary_metrics,
    log_odds_mean,
    position_batches,
)


def test_position_batches_use_deployment_tiers() -> None:
    lengths = [500] * 48 + [700] * 32 + [1_000] * 16
    assert [len(batch) for batch in position_batches(lengths)] == [48, 32, 16]


def test_reasoning_uses_frozen_tail_truncation() -> None:
    reasoning = "dropped-prefix" + "a" * REASONING_MAX_CHARS
    messages = [
        {"role": "assistant", "content": "answer", "reasoning": reasoning}
    ]
    prompt = add_reasoning("member", messages)

    assert reasoning[-REASONING_MAX_CHARS:] in prompt
    assert "dropped-prefix" not in prompt
    assert "<assistant_reasoning>" in prompt


def test_binary_metrics_report_ties_and_continuous_auroc() -> None:
    metrics = binary_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.1, 0.8, 0.7]),
    )

    assert metrics["auroc"] == 1.0
    assert metrics["unique_scores"] == 3
    assert metrics["duplicate_score_excess_rows"] == 1
    assert metrics["rows_participating_in_ties"] == 2


def test_log_odds_mean_is_symmetric_and_bounded() -> None:
    left = np.asarray([0.1, 0.8, 0.5])
    right = np.asarray([0.7, 0.2, 0.5])

    values = log_odds_mean(left, right)
    reverse = log_odds_mean(right, left)

    assert np.allclose(values, reverse)
    assert np.all((values > 0.0) & (values < 1.0))
