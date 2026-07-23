from experiments.liars_bench_distillation.analyze_soft_trigger_gemma_confirmation import analyze


def test_family_confirmation_gate_requires_generalization_and_preservation() -> None:
    external = {
        "conditions": {
            "baseline": {
                "metrics": {"balanced_accuracy": 0.80},
                "per_source_model": {"gemma-a": {"balanced_accuracy": 0.80}},
                "parse_errors": 1,
            },
            "candidate": {
                "metrics": {"balanced_accuracy": 0.82},
                "per_source_model": {"gemma-a": {"balanced_accuracy": 0.82}},
                "parse_errors": 2,
            },
        }
    }
    baseline_comp = {"per_dataset": {"dev-gemma-a": {"balanced_accuracy": 0.95}}}
    candidate_comp = {"per_dataset": {"dev-gemma-a": {"balanced_accuracy": 0.95}}}

    assert analyze(external, baseline_comp, candidate_comp)["gate"]["passed"] is True
    external["conditions"]["candidate"]["metrics"]["balanced_accuracy"] = 0.805
    assert analyze(external, baseline_comp, candidate_comp)["gate"]["passed"] is False
