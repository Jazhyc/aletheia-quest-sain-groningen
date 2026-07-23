import numpy as np
import pytest

from experiments.blackbox.analyze_generation_cap_prefix import (
    balanced_accuracy,
    evaluate_rating_matrix,
)


def test_rating_matrix_uses_max_and_reports_scenarios() -> None:
    ratings = np.asarray([[1, 1, 2, 1], [1, 3, 1, 1]])
    labels = np.asarray([0, 1, 1, 0])
    datasets = [
        "instructed-deception-a",
        "instructed-deception-a",
        "varied-deception-b",
        "varied-deception-b",
    ]

    result = evaluate_rating_matrix(ratings, labels, datasets)

    assert result["balanced_accuracy"] == 1.0
    assert result["macro_dataset_balanced_accuracy"] == 1.0
    assert result["scenario_balanced_accuracy"] == {
        "instructed": 1.0,
        "varied": 1.0,
    }


def test_balanced_accuracy_rejects_one_class() -> None:
    with pytest.raises(ValueError, match="both labels"):
        balanced_accuracy(np.asarray([1, 1]), np.asarray([1, 0]))
