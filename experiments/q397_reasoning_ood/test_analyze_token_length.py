"""Tests for the Q397 OOD token-length analysis."""

from __future__ import annotations

import pytest

from experiments.q397_reasoning_ood.analyze_token_length import (
    add_auroc_attribution,
    auroc,
)


def test_auroc_handles_ties() -> None:
    assert auroc([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) == pytest.approx(
        0.875
    )


def test_pair_attribution_averages_to_auroc_delta_by_label() -> None:
    rows = [
        {"label": 0, "direct_score": 0.4, "blend_score": 0.1},
        {"label": 0, "direct_score": 0.8, "blend_score": 0.2},
        {"label": 1, "direct_score": 0.5, "blend_score": 0.3},
        {"label": 1, "direct_score": 0.9, "blend_score": 0.4},
    ]
    expected_delta = auroc(
        [row["label"] for row in rows],
        [row["blend_score"] for row in rows],
    ) - auroc(
        [row["label"] for row in rows],
        [row["direct_score"] for row in rows],
    )

    add_auroc_attribution(rows)

    for label in (0, 1):
        label_mean = sum(
            row["auroc_pair_gain"] for row in rows if row["label"] == label
        ) / 2
        assert label_mean == pytest.approx(expected_delta)
