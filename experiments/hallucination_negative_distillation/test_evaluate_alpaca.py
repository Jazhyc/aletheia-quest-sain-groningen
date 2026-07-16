from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hallucination_negative_distillation.evaluate_alpaca import (
    alpaca_gate,
)


def test_alpaca_gate_requires_material_reduction_and_low_fpr() -> None:
    assert alpaca_gate(0.05, 0.02)["passed"]
    assert not alpaca_gate(0.04, 0.03)["passed"]
    assert not alpaca_gate(0.06, 0.03)["passed"]
