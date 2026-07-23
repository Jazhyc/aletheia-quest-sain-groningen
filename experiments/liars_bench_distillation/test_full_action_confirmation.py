from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.evaluate_full_action_confirmation import (
    action_gate,
    left_truncate_prompt_ids,
)
from experiments.liars_bench_distillation.evaluate_action_recipient_clarification import (
    recipient_clarification_gate,
)


def test_full_action_gate_requires_gain_absolute_ba_and_low_fpr() -> None:
    general = {"balanced_accuracy": 0.58, "fpr": 0.70}

    assert action_gate(
        general, {"balanced_accuracy": 0.75, "fpr": 0.08}
    )["passed"]
    assert not action_gate(
        general, {"balanced_accuracy": 0.67, "fpr": 0.08}
    )["passed"]
    assert not action_gate(
        general, {"balanced_accuracy": 0.75, "fpr": 0.20}
    )["passed"]


def test_left_truncate_prompt_ids_keeps_deployment_suffix() -> None:
    assert left_truncate_prompt_ids([1, 2, 3, 4], 3) == [2, 3, 4]
    assert left_truncate_prompt_ids([1, 2], 3) == [1, 2]


def test_recipient_clarification_gate_requires_broad_source_gain() -> None:
    baseline = {"balanced_accuracy": 0.70, "fpr": 0.04}
    candidate = {"balanced_accuracy": 0.80, "fpr": 0.10}
    baseline_by_source = {
        source: {"balanced_accuracy": 0.70} for source in "abcd"
    }
    passing = {
        "a": {"balanced_accuracy": 0.80},
        "b": {"balanced_accuracy": 0.79},
        "c": {"balanced_accuracy": 0.78},
        "d": {"balanced_accuracy": 0.69},
    }
    assert recipient_clarification_gate(
        baseline, candidate, baseline_by_source, passing
    )["passed"]

    narrow = {
        "a": {"balanced_accuracy": 0.90},
        "b": {"balanced_accuracy": 0.71},
        "c": {"balanced_accuracy": 0.69},
        "d": {"balanced_accuracy": 0.69},
    }
    assert not recipient_clarification_gate(
        baseline, candidate, baseline_by_source, narrow
    )["passed"]
