from __future__ import annotations

from experiments.liars_bench_distillation.analyze_soft_trigger_focus import (
    competition_balanced_accuracy,
    source_family,
    soft_trigger_gate,
)


def test_soft_trigger_gate_requires_target_gain_and_broad_preservation() -> None:
    assert soft_trigger_gate(
        0.90,
        0.899,
        0.89,
        0.90,
        {"gemma": 0.00, "llama": 0.01, "mistral": -0.02, "qwen": 0.01},
    )["passed"]
    assert not soft_trigger_gate(
        0.90,
        0.899,
        0.89,
        0.895,
        {"gemma": 0.00},
    )["passed"]


def test_soft_trigger_source_family_collapses_lora_names() -> None:
    assert source_family("gemma-3-27b-it-lora-greeting") == "gemma"
    assert source_family("llama-v3.3-70b-instruct") == "llama"
    assert source_family("mistral-small-3.1-24b-instruct") == "mistral"
    assert source_family("qwen-2.5-72b-it-lora-time") == "qwen"
    assert not soft_trigger_gate(
        0.90,
        0.899,
        0.89,
        0.91,
        {"gemma": -0.04, "qwen": 0.04},
    )["passed"]


def test_competition_balanced_accuracy_reads_current_nested_schema() -> None:
    assert competition_balanced_accuracy({
        "metrics": {"all": {"balanced_accuracy": 0.9}}
    }) == 0.9
