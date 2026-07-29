from __future__ import annotations

import math

from experiments.q397_base_prompt_ensemble.analyze import (
    logit,
    paired_rows,
    sigmoid,
)


def test_logit_sigmoid_round_trip() -> None:
    for value in (0.01, 0.2, 0.5, 0.8, 0.99):
        assert math.isclose(sigmoid(logit(value)), value, abs_tol=1.0e-12)


def test_paired_rows_uses_frozen_sixty_forty_log_odds() -> None:
    key = ("dataset", "1")
    q397 = {
        key: {
            "dataset": key[0],
            "index": key[1],
            "label": 1,
            "_score": 0.8,
        }
    }
    base = {
        key: {
            "dataset": key[0],
            "index": key[1],
            "label": 1,
            "_score": 0.2,
        }
    }
    rows, baseline, candidate = paired_rows(q397, base)
    assert rows[0]["label"] == 1
    assert baseline == [0.8]
    assert math.isclose(candidate[0], sigmoid(0.2 * logit(0.8)), abs_tol=1e-12)
