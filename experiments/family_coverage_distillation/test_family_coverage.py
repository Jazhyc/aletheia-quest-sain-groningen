from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.family_coverage_distillation.analyze import analyze


def make_rows(prediction: int) -> dict[tuple[str, str], dict[str, object]]:
    rows = {}
    datasets = (
        "dev-varied-deception-Qwen-model",
        "dev-instructed-deception-gemma-model",
        "dev-instructed-deception-NVIDIA-Nemotron-model",
    )
    for dataset in datasets:
        for index, label in enumerate((0, 1)):
            rows[(dataset, str(index))] = {
                "dataset": dataset,
                "index": str(index),
                "label": label,
                "prediction": prediction,
            }
    return rows


def test_acceptance_requires_nonregressing_non_qwen_families() -> None:
    baseline = make_rows(0)
    candidate = make_rows(0)
    candidate[("dev-instructed-deception-gemma-model", "1")]["prediction"] = 1

    result = analyze(baseline, candidate)

    assert result["accepted"] is True
    assert result["deltas"]["Gemma"] == 0.5
    assert result["fixes"] == 1
    assert result["breaks"] == 0
