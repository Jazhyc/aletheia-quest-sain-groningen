import numpy as np
import pandas as pd

from experiments.heterogeneous_adapter_ensemble import evaluate_ngram_stack


def test_build_report_labels_frozen_evaluation_split(monkeypatch) -> None:
    evaluation = pd.DataFrame(
        {
            "dataset": [
                "dev-instructed-a",
                "dev-instructed-a",
                "dev-varied-deception-b",
                "dev-varied-deception-b",
            ],
            "label": [0, 1, 0, 1],
            "deception": [0.0, 1.0, 0.0, 0.0],
        }
    )
    scores = {
        "adapter_only": np.array([0.1, 0.9, 0.2, 0.4]),
        "ngram_only": np.array([0.2, 0.8, 0.7, 0.6]),
        "hybrid": np.array([0.1, 0.8, 0.2, 0.7]),
    }
    monkeypatch.setattr(
        evaluate_ngram_stack,
        "fit_scores",
        lambda train, candidate, names: (scores, {"frozen": True}),
    )

    report = evaluate_ngram_stack.build_report(
        pd.DataFrame(),
        evaluation,
        ["deception", "incorrectness", "resolved_intent"],
        split="test",
        excluded_rows=3,
        max_context_chars=3000,
    )

    assert report["protocol"]["fit_split"] == "train"
    assert report["protocol"]["evaluation_split"] == "test"
    assert report["test"]["hybrid"]["balanced_accuracy"] == 1.0
    assert report["test_comparison"]["hybrid"]["versus_deception"] == {
        "fixes": 1,
        "breaks": 0,
        "decision_disagreements": 1,
    }
