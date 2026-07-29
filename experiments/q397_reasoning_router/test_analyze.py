from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.q397_reasoning_router.analyze import (
    Gate,
    add_features,
    gate_scores,
    logit,
    macro_auroc,
    pairwise_complementarity,
    sigmoid,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": ["a"] * 4 + ["b"] * 4,
        "label": [0, 0, 1, 1] * 2,
        "direct_score": [0.1, 0.2, 0.8, 0.9, 0.2, 0.7, 0.3, 0.8],
        "post_reasoning_score": [
            0.2, 0.3, 0.7, 0.8, 0.1, 0.2, 0.8, 0.9,
        ],
        "reasoning_reply": ["Prediction:0"] * 8,
        "reasoning_tokens": [20, 30, 40, 50, 20, 30, 40, 50],
    })


def test_logit_round_trip() -> None:
    values = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    assert np.allclose(sigmoid(logit(values)), values)


def test_macro_auroc_averages_dataset_units() -> None:
    frame = sample_frame()
    assert macro_auroc(frame, frame["direct_score"].to_numpy()) == 0.875
    assert macro_auroc(frame, frame["post_reasoning_score"].to_numpy()) == 1.0


def test_gate_selects_scores_rowwise() -> None:
    frame = add_features(sample_frame())
    gate = Gate(
        "short",
        lambda rows: rows["reasoning_tokens"].le(30).to_numpy(),
    )
    expected = np.array([0.2, 0.3, 0.8, 0.9, 0.1, 0.2, 0.3, 0.8])
    assert np.allclose(gate_scores(frame, gate), expected)


def test_pairwise_complementarity_counts_all_pairs() -> None:
    report = pairwise_complementarity(sample_frame())
    assert sum(report.values()) == 8
    assert report["reasoning_recovers_direct_pair"] == 1
    assert report["reasoning_harms_direct_pair"] == 0
