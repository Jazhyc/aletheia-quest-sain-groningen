from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hallucination_negative_distillation.screen_alpaca import (
    select_per_model,
)


def test_select_per_model_is_stable_and_bounded() -> None:
    rows = [
        {"index": f"{model}:{index}", "source_model": model}
        for model in ("a", "b")
        for index in range(5)
    ]

    first = select_per_model(rows, maximum_per_model=2, seed=7)

    assert len(first) == 4
    assert sum(row["source_model"] == "a" for row in first) == 2
    assert sum(row["source_model"] == "b" for row in first) == 2
    assert first == select_per_model(rows, maximum_per_model=2, seed=7)
    assert first != select_per_model(rows, maximum_per_model=2, seed=8)
