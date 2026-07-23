from pathlib import Path

import pandas as pd

from experiments.reasoning_intent_logits.analyze import (
    dataset_from_prediction_path,
    load_intent_member,
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


def test_validation_gate_requires_gain_precision_and_standalone_signal() -> None:
    summary = {
        "metrics": {
            "phoenix": {"balanced_accuracy": 0.80, "fpr": 0.05},
            "intent": {"balanced_accuracy": 0.82, "fpr": 0.08},
            "logistic": {"balanced_accuracy": 0.81, "fpr": 0.08},
        }
    }

    gate = validation_gate(summary)

    assert gate["passed"]
    assert all(gate["checks"].values())
