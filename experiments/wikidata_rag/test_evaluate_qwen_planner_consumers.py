from __future__ import annotations

import pandas as pd
import pytest

from experiments.wikidata_rag.evaluate_qwen_planner_consumers import (
    condition_uplift,
    parse_named_path,
    replace_active_scores,
    scoring_prompt,
)


def test_parse_named_path_requires_a_name_and_location() -> None:
    name, path = parse_named_path("consumer=results/adapter")

    assert name == "consumer"
    assert path.name == "adapter"
    with pytest.raises(ValueError):
        parse_named_path("results/adapter")


def test_scoring_prompt_supports_both_frozen_label_boundaries() -> None:
    assert scoring_prompt("chat", "direct") == "chatPrediction:"
    assert "<reasoning_summary>" in scoring_prompt("chat", "empty_summary")


def test_replace_active_scores_preserves_inactive_rows() -> None:
    baseline = pd.DataFrame([
        {"dataset": "d", "index": 0, "label": 0, "score": 0.1},
        {"dataset": "d", "index": 1, "label": 1, "score": 0.8},
    ])
    active = pd.DataFrame([
        {
            "dataset": "d",
            "index": 1,
            "label": 1,
            "empty_score": 0.8,
            "real_score": 0.9,
        }
    ])

    result = replace_active_scores(baseline, active, "real_score").set_index("index")

    assert result.loc[0, "score"] == 0.1
    assert result.loc[1, "score"] == 0.9


def test_condition_uplift_is_scenario_specific() -> None:
    empty = {
        "metrics": {
            "all": {"auroc": 0.8},
            "instructed": {"auroc": 0.9},
            "varied": {"auroc": 0.7},
        }
    }
    real = {
        "metrics": {
            "all": {"auroc": 0.82},
            "instructed": {"auroc": 0.9},
            "varied": {"auroc": 0.75},
        }
    }

    assert condition_uplift(empty, real) == {
        "all": pytest.approx(0.02),
        "instructed": 0.0,
        "varied": pytest.approx(0.05),
    }
