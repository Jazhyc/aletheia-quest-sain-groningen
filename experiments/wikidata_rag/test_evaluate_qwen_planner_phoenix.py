import pandas as pd
import pytest

from experiments.wikidata_rag.evaluate_qwen_planner_phoenix import (
    compose_conditions,
    mean_logodds,
)


def test_mean_logodds_preserves_equal_values_and_is_symmetric() -> None:
    assert mean_logodds(0.2, 0.2) == pytest.approx(0.2)
    assert mean_logodds(0.2, 0.8) == pytest.approx(0.5)
    assert mean_logodds(0.8, 0.2) == pytest.approx(0.5)


def test_compose_conditions_updates_only_active_rows() -> None:
    baseline = pd.DataFrame([
        {"dataset": "a", "index": 1, "label": 0, "score": 0.1},
        {"dataset": "b", "index": 2, "label": 1, "score": 0.8},
    ])
    active = pd.DataFrame([{
        "dataset": "a",
        "index": 1,
        "label": 0,
        "empty_score": 0.2,
        "real_score": 0.4,
        "shuffled_score": 0.3,
    }])

    conditions = compose_conditions(baseline, active)

    assert conditions["baseline"]["score"].tolist() == [0.1, 0.8]
    assert conditions["recomputed_empty"]["score"].tolist() == [0.2, 0.8]
    assert conditions["real_replace"]["score"].tolist() == [0.4, 0.8]
    assert conditions["shuffled_replace"]["score"].tolist() == [0.3, 0.8]
    assert conditions["real_blend"]["score"].iloc[0] == pytest.approx(
        mean_logodds(0.1, 0.4)
    )
    assert conditions["shuffled_blend"]["score"].iloc[0] == pytest.approx(
        mean_logodds(0.1, 0.3)
    )
