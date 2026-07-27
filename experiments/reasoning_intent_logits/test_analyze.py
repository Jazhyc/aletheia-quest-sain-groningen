from pathlib import Path

import pandas as pd

from experiments.reasoning_intent_logits.analyze import (
    dataset_from_prediction_path,
    load_intent_member,
    load_phoenix,
    score_bundle,
    score_diagnostics,
    validation_gate,
)


def test_dataset_from_prediction_path_restores_org_separator() -> None:
    path = Path(
        "aletheias-quest__dev-varied-deception-Qwen3.5-27B-None.csv"
    )
    assert dataset_from_prediction_path(path) == (
        "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None"
    )


def test_load_intent_member_keeps_only_varied_predictions(tmp_path: Path) -> None:
    member = tmp_path / "resolved_intent"
    member.mkdir()
    pd.DataFrame({"index": [1], "deceptive": [True], "score": [0.8]}).to_csv(
        member / "aletheias-quest__dev-varied-deception-Qwen.csv",
        index=False,
    )
    pd.DataFrame({"index": [2], "deceptive": [False], "score": [0.2]}).to_csv(
        member / "aletheias-quest__dev-instructed-deception-Qwen.csv",
        index=False,
    )

    frame = load_intent_member(tmp_path, "resolved_intent")

    assert frame[["dataset", "index", "intent_score"]].to_dict("records") == [{
        "dataset": "aletheias-quest/dev-varied-deception-Qwen",
        "index": "1",
        "intent_score": 0.8,
    }]


def test_load_phoenix_selects_direct_margin_and_varied_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "phoenix.jsonl"
    pd.DataFrame([
        {
            "dataset": "dev-varied-deception-Qwen",
            "index": 1,
            "label": 1,
            "score": 0.0,
            "direct_margin_score": 0.8,
        },
        {
            "dataset": "dev-instructed-deception-Qwen",
            "index": 2,
            "label": 0,
            "score": 1.0,
            "direct_margin_score": 0.2,
        },
    ]).to_json(path, orient="records", lines=True)

    frame = load_phoenix(path, "direct_margin_score")

    assert frame[["dataset", "index", "phoenix_score"]].to_dict("records") == [{
        "dataset": "dev-varied-deception-Qwen",
        "index": "1",
        "phoenix_score": 0.8,
    }]
    assert not frame["phoenix_parse_error"].any()


def test_score_bundle_includes_fixed_continuous_blends() -> None:
    frame = pd.DataFrame({
        "dataset": ["unit", "unit"],
        "phoenix_score": [0.2, 0.8],
        "intent_score": [0.8, 0.2],
    })

    scores = score_bundle(frame, logistic_scores=pd.Series([0.4, 0.6]).to_numpy())

    assert scores["mean_probability"].tolist() == [0.5, 0.5]
    assert scores["mean_log_odds"].tolist() == [0.5, 0.5]
    assert scores["maximum"].tolist() == [0.8, 0.8]
    assert scores["rank_average"].tolist() == [0.75, 0.75]


def test_score_diagnostics_reports_cross_label_ties() -> None:
    frame = pd.DataFrame({
        "dataset": ["a", "a", "b", "b"],
        "label": [0, 1, 0, 1],
    })

    diagnostics = score_diagnostics(frame, pd.Series([0.2, 0.2, 0.1, 0.9]))

    assert diagnostics["unique_scores"] == 3
    assert diagnostics["duplicate_row_fraction"] == 0.25
    assert diagnostics["macro_cross_label_pair_fraction"] == 0.5


def test_validation_gate_requires_auroc_gain_and_standalone_signal() -> None:
    summary = {
        "metrics": {
            "phoenix": {"auroc": 0.80, "fpr": 0.05},
            "intent": {"auroc": 0.82, "fpr": 0.08},
            "mean_log_odds": {"auroc": 0.81, "fpr": 0.08},
            "logistic": {"auroc": 0.815, "fpr": 0.08},
        }
    }

    gate = validation_gate(summary)

    assert gate["passed"]
    assert all(gate["checks"].values())
