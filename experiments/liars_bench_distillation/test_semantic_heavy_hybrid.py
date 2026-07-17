from experiments.liars_bench_distillation.analyze_semantic_heavy_hybrid import (
    SPECIALIST_BY_CATEGORY,
    analyze,
)


def test_hybrid_routes_only_explicit_report_contracts() -> None:
    assert SPECIALIST_BY_CATEGORY == {
        "harm-pressure-knowledge-report": "epistemic",
        "insider-trading": "action",
    }
    assert "harm-pressure-choice" not in SPECIALIST_BY_CATEGORY
    assert "soft-trigger" not in SPECIALIST_BY_CATEGORY


def test_fixed_semantic_routes_can_improve_report_modes_without_touching_others() -> None:
    categories = (
        "harm-pressure-choice",
        "harm-pressure-knowledge-report",
        "insider-trading",
        "soft-trigger",
    )
    heavy = []
    specialists = {"harm-pressure-knowledge-report": [], "insider-trading": []}
    for category in categories:
        for offset in range(200):
            label = offset % 2
            index = f"{category}:{offset}"
            heavy.append({
                "index": index,
                "category": category,
                "source_family": "qwen",
                "label": label,
                "prediction": label if category not in specialists else 0,
            })
            if category in specialists:
                specialists[category].append({
                    "index": index,
                    "label": label,
                    "prediction": label,
                })

    result = analyze(
        heavy,
        specialists["harm-pressure-knowledge-report"],
        specialists["insider-trading"],
    )

    assert result["gate"]["passed"] is True
    assert result["changed"] == result["fixes"] == 200
    assert result["breaks"] == 0
    assert result["paired_bootstrap"]["macro"]["lower_95"] > 0
