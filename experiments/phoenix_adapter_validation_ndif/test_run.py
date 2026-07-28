from __future__ import annotations

import numpy as np
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_adapter_validation_ndif.run import (
    binary_metrics,
    paired_report,
    position_batches,
)


def test_position_batches_respect_frozen_length_tiers() -> None:
    lengths = [500] * 48 + [700] * 32 + [1_000] * 16
    assert [len(batch) for batch in position_batches(lengths)] == [48, 32, 16]


def test_binary_metrics_reports_continuous_auroc() -> None:
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.2, 0.7])
    metrics = binary_metrics(labels, scores)

    assert metrics["auroc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["unique_scores"] == 4


def test_paired_report_detects_active_adapter() -> None:
    base = np.array([0.1, 0.2, 0.3])
    adapted = np.array([0.2, 0.1, 0.4])
    report = paired_report(base, adapted)

    assert report["exact_equal"] == 0
    assert report["mean_absolute_difference"] > 0
