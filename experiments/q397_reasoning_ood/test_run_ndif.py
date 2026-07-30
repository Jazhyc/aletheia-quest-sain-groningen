from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.q397_reasoning_ood.run_ndif import (
    compare_conditions,
    summarize_condition,
)


def test_summarize_condition_excludes_single_class_category_from_macro() -> None:
    records = pd.DataFrame({
        "category": ["honest", "honest", "mixed", "mixed"],
        "label": [0, 0, 0, 1],
    })
    summary = summarize_condition(
        records,
        np.asarray([0.1, 0.2, 0.3, 0.9]),
    )

    assert summary["macro_auroc"] == 1.0
    assert summary["auroc_categories"] == 1
    assert summary["categories"]["honest"]["auroc"] is None
    assert summary["categories"]["mixed"]["auroc"] == 1.0


def test_compare_conditions_reports_category_direction() -> None:
    direct = {
        "macro_auroc": 0.6,
        "categories": {
            "a": {"auroc": 0.5},
            "b": {"auroc": 0.7},
            "honest": {"auroc": None},
        },
    }
    blend = {
        "macro_auroc": 0.65,
        "categories": {
            "a": {"auroc": 0.7},
            "b": {"auroc": 0.6},
            "honest": {"auroc": None},
        },
    }

    comparison = compare_conditions(direct, blend)

    assert np.isclose(comparison["macro_auroc_delta"], 0.05)
    assert comparison["category_wins"] == 1
    assert comparison["category_losses"] == 1
    assert comparison["category_ties"] == 0
    assert comparison["per_category_delta"] == {
        "a": 0.19999999999999996,
        "b": -0.09999999999999998,
    }
