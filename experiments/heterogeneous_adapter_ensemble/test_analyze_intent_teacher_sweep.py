from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.heterogeneous_adapter_ensemble.analyze_intent_teacher_sweep import (
    candidate_passes_gate,
)


BASELINE = {
    "balanced_accuracy": 0.84,
    "recall": 0.72,
    "fpr": 0.04,
}


def test_candidate_passes_all_frozen_checks() -> None:
    checks = candidate_passes_gate(
        BASELINE,
        {"balanced_accuracy": 0.85, "recall": 0.715, "fpr": 0.02},
        {"fixes": 12, "breaks": 5, "decision_disagreements": 17},
    )

    assert checks["passed"] is True


def test_candidate_fails_for_recall_loss_or_unfavorable_changes() -> None:
    checks = candidate_passes_gate(
        BASELINE,
        {"balanced_accuracy": 0.85, "recall": 0.70, "fpr": 0.02},
        {"fixes": 6, "breaks": 4, "decision_disagreements": 10},
    )

    assert checks["recall_preserved"] is False
    assert checks["fix_break_ratio"] is False
    assert checks["passed"] is False
