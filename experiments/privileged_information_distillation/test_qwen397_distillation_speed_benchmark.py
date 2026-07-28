from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT
    / "experiments/privileged_information_distillation"
    / "run_qwen397_distillation_speed_benchmark_lambda.sh"
)
SUMMARY_PATH = (
    ROOT
    / "experiments/privileged_information_distillation"
    / "summarize_qwen397_speed_benchmark.py"
)
TRAINER_PATH = (
    ROOT
    / "experiments/privileged_information_distillation"
    / "train_student_sft.py"
)


def load_summary_module():
    spec = importlib.util.spec_from_file_location("q397_speed_summary", SUMMARY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_preserves_effective_batch_and_tests_compile() -> None:
    source = LAUNCHER.read_text()

    assert "batch_sizes=(2 4 8 16)" in source
    assert "accumulations=(16 8 4 2)" in source
    assert "student.training.torch_compile=${compile}" in source
    assert "--best-shell" in source
    assert 'compile_name="b${best_batch}_e32_compile"' in source
    assert "benchmark condition failed; continuing" in source
    assert "compile benchmark failed; retaining eager results" in source


def test_eager_training_does_not_pass_non_null_compile_options() -> None:
    source = TRAINER_PATH.read_text()

    assert "torch_compile=torch_compile" in source
    assert "if torch_compile\n            else None" in source


def test_summary_selects_highest_sample_throughput(tmp_path: Path) -> None:
    module = load_summary_module()
    (tmp_path / "b2_e32_eager.out").write_text(
        "{'train_runtime': 10.0, 'train_samples_per_second': 12.0, "
        "'train_steps_per_second': 0.4}\n"
    )
    (tmp_path / "b8_e32_eager.out").write_text(
        "{'train_runtime': 8.0, 'train_samples_per_second': 15.0, "
        "'train_steps_per_second': 0.5}\n"
    )

    rows = [
        module.load_row(tmp_path, "b2_e32_eager"),
        module.load_row(tmp_path, "b8_e32_eager"),
    ]
    best = sorted(rows, key=lambda row: -row["train_samples_per_second"])[0]

    assert best["batch_size"] == 8
    assert best["gradient_accumulation_steps"] == 4
    with pytest.raises(ValueError, match="unexpected benchmark condition"):
        module.load_row(tmp_path, "bad")
