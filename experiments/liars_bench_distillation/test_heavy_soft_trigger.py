from experiments.liars_bench_distillation.evaluate_heavy_soft_trigger import (
    predict_with_family_cutoff,
    source_family,
)


def test_frozen_gemma_cutoff_only_changes_family_parameter() -> None:
    assert predict_with_family_cutoff(6, "gemma") == 0
    assert predict_with_family_cutoff(7, "gemma") == 1
    assert predict_with_family_cutoff(2, "qwen") == 1
    assert predict_with_family_cutoff(1, "qwen") == 0


def test_source_family_collapses_base_and_lora_names() -> None:
    assert source_family("gemma-3-27b-it-lora-time") == "gemma"
    assert source_family("llama-v3.3-70b-instruct") == "llama"
    assert source_family("mistral-small-3.1-24b-instruct") == "mistral"
    assert source_family("qwen-2.5-72b-it-lora-gender") == "qwen"
