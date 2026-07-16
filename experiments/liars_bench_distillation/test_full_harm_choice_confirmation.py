from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.evaluate_full_harm_choice_confirmation import (
    harm_choice_gate,
)


def test_full_harm_choice_gate_requires_gain_absolute_ba_and_low_fpr() -> None:
    general = {"balanced_accuracy": 0.58, "fpr": 0.40}

    assert harm_choice_gate(
        general, {"balanced_accuracy": 0.75, "fpr": 0.08}
    )["passed"]
    assert not harm_choice_gate(
        general, {"balanced_accuracy": 0.67, "fpr": 0.08}
    )["passed"]
    assert not harm_choice_gate(
        general, {"balanced_accuracy": 0.75, "fpr": 0.20}
    )["passed"]
