import json
from pathlib import Path

import yaml

from experiments.privileged_information_distillation.build_kimi_pairwise_holdout import (
    build,
)
from experiments.privileged_information_distillation.select_kimi_pairwise_scale import (
    select_scale,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT
    / "experiments"
    / "privileged_information_distillation"
    / "run_kimi_k3_pairwise_scale_sweep_lambda.sh"
)


def test_holdout_builder_stratifies_every_dataset_and_label(tmp_path: Path) -> None:
    student_rows = tmp_path / "student_rows.jsonl"
    rows = [
        {
            "dataset": dataset,
            "index": f"{dataset}-{label}-{index}",
            "label": label,
        }
        for dataset in ("dataset-a", "dataset-b")
        for label in (0, 1)
        for index in range(10)
    ]
    student_rows.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    output = tmp_path / "holdout"
    audit = build(
        student_rows,
        output,
        train_fraction=0.8,
        seed=17,
        expected_rows=40,
    )

    assert audit["train_rows"] == 32
    assert audit["holdout_rows"] == 8
    split = yaml.safe_load((output / "holdout_splits" / "dry.train.yaml").read_text())
    assert len(split["datasets"]) == 2
    manifest = [
        json.loads(line)
        for line in (output / "train_manifest.jsonl").read_text().splitlines()
    ]
    assert len(manifest) == 32


def write_result(adapter: Path, run_name: str, auroc: float) -> None:
    output = adapter.parent / run_name
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps({
        "score_metrics": {
            "direct_margin": {
                "all": {"auroc": auroc},
            },
        },
    }))


def test_scale_selector_averages_orders_and_breaks_near_ties_conservatively(
    tmp_path: Path,
) -> None:
    candidates = []
    scores = {
        0.0: (0.8980, 0.8980),
        0.1: (0.8998, 0.9002),
        0.3: (0.9007, 0.9003),
        1.0: (0.8950, 0.8960),
    }
    for weight, values in scores.items():
        adapter = tmp_path / f"weight-{weight}" / "adapter"
        adapter.mkdir(parents=True)
        write_result(adapter, "forward", values[0])
        write_result(adapter, "reverse", values[1])
        candidates.append((weight, adapter))

    selected = select_scale(
        candidates,
        ["forward", "reverse"],
        tie_tolerance=0.001,
    )

    assert selected["best_mean_auroc"] == 0.9005
    assert selected["selected_weight"] == 0.1
    assert selected["selected_by_conservative_tie_break"] is True


def test_scale_sweep_runner_uses_internal_selection_before_validation() -> None:
    source = RUNNER.read_text()

    assert "WEIGHTS=(0.0 0.1 0.3 1.0)" in source
    assert "--train-fraction 0.8" in source
    assert "paired_batching_mode=same_dataset" in source
    assert "--run-name internal_holdout_forward" in source
    assert "--run-name internal_holdout_reverse" in source
    assert "--tie-tolerance 0.001" in source
    assert "validation_grouped_scale_forward" in source
    assert "validation_grouped_scale_reverse" in source
