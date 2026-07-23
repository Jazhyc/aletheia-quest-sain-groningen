from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reasoning_ngram_ensemble.analyze_ensemble import (
    FEATURES,
    grouped_oof_scores,
    load_joined,
    nested_grouped_oof_scores,
    or_scores,
)


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for group_number, dataset in enumerate(("a", "b", "c")):
        for number in range(8):
            label = number % 2
            rows.append({
                "dataset": dataset,
                "index": f"{dataset}-{number}",
                "label": label,
                "phoenix_score": float(label if number != group_number else 1 - label),
                "reasoning_ngram_score": 0.8 if label else 0.2,
            })
    return pd.DataFrame(rows)


def test_or_is_max_score_and_logical_or_at_threshold() -> None:
    frame = pd.DataFrame({
        "phoenix_score": [0.0, 0.0, 1.0, 1.0],
        "reasoning_ngram_score": [0.1, 0.8, 0.2, 0.9],
    })

    scores = or_scores(frame)

    assert np.allclose(scores, [0.1, 0.8, 1.0, 1.0])
    assert (scores >= 0.5).tolist() == [False, True, True, True]


def test_grouped_oof_scores_every_row() -> None:
    frame = synthetic_frame()

    scores = grouped_oof_scores(frame, c=0.1)

    assert scores.shape == (len(frame),)
    assert np.isfinite(scores).all()
    assert ((0.0 <= scores) & (scores <= 1.0)).all()


def test_nested_grouped_oof_selects_without_scoring_fit_rows() -> None:
    frame = synthetic_frame()

    scores, selections = nested_grouped_oof_scores(frame)

    assert np.isfinite(scores).all()
    assert set(selections) == {"a", "b", "c"}


def test_load_joined_rejects_missing_member_rows(tmp_path: Path) -> None:
    phoenix = pd.DataFrame({
        "dataset": ["dev-varied-deception-x"] * 2,
        "index": [1, 2],
        "label": [0, 1],
        "score": [0.0, 1.0],
    })
    ngram = pd.DataFrame({
        "dataset": ["dev-varied-deception-x"],
        "index": [1],
        "label": [0],
        "score": [0.1],
    })
    phoenix_path = tmp_path / "phoenix.jsonl"
    ngram_path = tmp_path / "ngram.csv"
    phoenix.to_json(phoenix_path, orient="records", lines=True)
    ngram.to_csv(ngram_path, index=False)

    try:
        load_joined(phoenix_path, ngram_path)
    except ValueError as error:
        assert "member key mismatch" in str(error)
    else:
        raise AssertionError("expected mismatched member keys to fail")


def test_feature_order_is_frozen() -> None:
    assert FEATURES == ["phoenix_score", "reasoning_ngram_score"]
