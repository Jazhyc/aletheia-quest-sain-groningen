from __future__ import annotations

import pandas as pd

from experiments.wikidata_rag.analyze_phoenix_candidate_oracle import (
    best_for_label,
    compose_oracles,
    row_oracle_scores,
)


def test_best_for_label_moves_scores_in_the_correct_direction() -> None:
    assert best_for_label([0.2, 0.7, 0.4], 1) == (0.7, 1)
    assert best_for_label([0.2, 0.7, 0.1], 0) == (0.1, 2)


def test_semantic_retrieval_oracle_does_not_peek_at_deception_label() -> None:
    source = {
        "empty_score": 0.4,
        "label": 0,
        "candidates": [
            {"score": 0.8, "semantic_label": "decisive"},
            {"score": 0.1, "semantic_label": "decisive"},
        ],
    }
    scores = row_oracle_scores(source)

    assert scores["semantic_retrieval_oracle"] == 0.8
    assert scores["semantic_utility_oracle"] == 0.1


def test_compose_oracles_preserves_rows_outside_candidate_cache() -> None:
    baseline = pd.DataFrame([
        {"dataset": "d", "index": 0, "label": 1, "score": 0.3},
        {"dataset": "d", "index": 1, "label": 0, "score": 0.2},
    ])
    rows = [{
        "dataset": "d",
        "index": 0,
        "label": 1,
        "empty_score": 0.4,
        "candidates": [{"score": 0.9, "semantic_label": "decisive"}],
    }]

    conditions, counts = compose_oracles(baseline, rows)

    assert counts["rows_with_decisive"] == 1
    oracle = conditions["any_candidate_utility_oracle"].set_index("index")
    assert oracle.loc[0, "score"] == 0.9
    assert oracle.loc[1, "score"] == 0.2
