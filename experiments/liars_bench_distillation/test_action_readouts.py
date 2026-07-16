from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.evaluate_action_readouts import (
    readout_gate,
)


def test_action_readout_gate_requires_gain_absolute_ba_and_low_fpr() -> None:
    summary = {"balanced_accuracy": 0.76, "fpr": 0.06}

    assert readout_gate(
        summary, {"balanced_accuracy": 0.80, "fpr": 0.12}
    )["passed"]
    assert not readout_gate(
        summary, {"balanced_accuracy": 0.78, "fpr": 0.10}
    )["passed"]
    assert not readout_gate(
        summary, {"balanced_accuracy": 0.80, "fpr": 0.20}
    )["passed"]
