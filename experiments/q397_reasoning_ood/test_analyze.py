from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.q397_reasoning_ood.analyze import (
    log_odds_blend,
    paired_bootstrap,
)


def test_paired_bootstrap_excludes_single_class_and_finds_clear_gain() -> None:
    frame = pd.DataFrame({
        "category": ["mixed"] * 6 + ["honest"] * 2,
        "label": [0, 0, 0, 1, 1, 1, 0, 0],
        "direct_score": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2],
        "candidate_score": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.1, 0.2],
    })

    result = paired_bootstrap(
        frame,
        "candidate_score",
        samples=100,
        seed=7,
    )

    assert np.isclose(result["point_macro_delta"], 1.0)
    assert list(result["categories"]) == ["mixed"]
    assert result["macro_probability_positive"] == 1.0


def test_log_odds_blend_preserves_equal_scores() -> None:
    scores = pd.Series([0.1, 0.5, 0.9])

    blended = log_odds_blend(scores, scores)

    assert np.allclose(blended, scores.to_numpy())
