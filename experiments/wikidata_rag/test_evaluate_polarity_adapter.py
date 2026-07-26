import numpy as np

from experiments.wikidata_rag.evaluate_polarity_adapter import metrics, polar_rows


def test_polar_rows_drop_abstentions() -> None:
    rows = [
        {"annotation": {"polarity": "supports"}},
        {"annotation": {"polarity": "contradicts"}},
        {"annotation": {"polarity": None}},
    ]

    assert len(polar_rows(rows)) == 2


def test_metrics_use_contradiction_as_positive_class() -> None:
    report = metrics(np.asarray([0, 1]), np.asarray([0.1, 0.9]))

    assert report["auroc"] == 1.0
    assert report["contradictions"] == 1
