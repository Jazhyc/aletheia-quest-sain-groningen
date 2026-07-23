from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.model_specific_readout_routing.analyze import (
    apply_routes,
    organism_key,
    select_routes,
)


def test_organism_key_removes_scenario_only() -> None:
    assert organism_key(
        "aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo"
    ) == "Qwen3.5-27B-a-mo"


def test_select_routes_requires_material_summary_gain() -> None:
    rows = [
        {"dataset": "aletheias-quest/dev-instructed-deception-model-a", "label": label}
        for label in (0, 0, 1, 1)
    ] + [
        {"dataset": "aletheias-quest/dev-instructed-deception-model-b", "label": label}
        for label in (0, 0, 1, 1)
    ]
    summary = [0, 0, 1, 1, 0, 0, 1, 1]
    maximum = [1, 0, 1, 1, 1, 0, 1, 1]

    routes = select_routes(
        rows, summary, maximum, minimum_train_gain=0.1
    )

    assert routes == {"model-a": "summary", "model-b": "summary"}
    assert apply_routes(rows, summary, maximum, routes) == summary
