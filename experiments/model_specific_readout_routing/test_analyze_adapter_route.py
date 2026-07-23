from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.model_specific_readout_routing.analyze_adapter_route import (
    family_from_dataset,
    route_predictions,
)


def test_family_from_dataset_is_case_insensitive() -> None:
    assert family_from_dataset("dev-varied-deception-Qwen3.5-27B-lora") == "Qwen"
    assert family_from_dataset("dev-instructed-deception-gemma-3-27b") == "Gemma"
    assert family_from_dataset("dev-instructed-deception-NVIDIA-Nemotron") == "Nemotron"


def test_route_uses_varied_for_qwen_and_mixed_elsewhere() -> None:
    qwen = ("dev-varied-deception-Qwen3.5-27B-lora", "1")
    gemma = ("dev-instructed-deception-gemma-3-27b", "2")
    varied_rows = {
        qwen: {"dataset": qwen[0], "index": qwen[1], "label": 1, "prediction": 1},
        gemma: {"dataset": gemma[0], "index": gemma[1], "label": 1, "prediction": 0},
    }
    mixed_rows = {
        qwen: {**varied_rows[qwen], "prediction": 0},
        gemma: {**varied_rows[gemma], "prediction": 1},
    }

    rows, varied, mixed, routed = route_predictions(varied_rows, mixed_rows)

    predictions = {
        str(row["dataset"]): prediction for row, prediction in zip(rows, routed)
    }
    assert predictions[qwen[0]] == 1
    assert predictions[gemma[0]] == 1
    assert sorted(varied) == [0, 1]
    assert sorted(mixed) == [0, 1]
