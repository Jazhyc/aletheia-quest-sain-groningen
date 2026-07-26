from __future__ import annotations

from experiments.liars_bench_distillation.evaluate_action_report_auroc import (
    left_truncate_prompt_ids,
    load_selected,
    select_margin_arm,
)


def condition(
    auroc: float,
    *,
    llama: float,
    mistral: float,
) -> dict:
    return {
        "metrics": {"auroc": auroc},
        "per_source_model": {
            "llama": {"auroc": llama},
            "mistral": {"auroc": mistral},
        },
    }


def test_selection_requires_gain_and_source_preservation() -> None:
    summaries = {
        "phoenix_control": condition(0.60, llama=0.60, mistral=0.60),
        "action_direct": condition(0.68, llama=0.68, mistral=0.58),
        "action_empty": condition(0.70, llama=0.72, mistral=0.50),
        "action_reasoning": condition(0.64, llama=0.64, mistral=0.64),
    }

    selection = select_margin_arm(
        summaries,
        minimum_auroc_gain=0.05,
        maximum_source_auroc_loss=0.05,
    )

    assert selection["selected"] == "action_direct"
    assert selection["comparisons"]["action_direct"]["passes"]
    assert not selection["comparisons"]["action_empty"]["passes"]
    assert not selection["comparisons"]["action_reasoning"]["aggregate_passes"]


def test_selection_can_reject_every_arm() -> None:
    summaries = {
        "phoenix_control": condition(0.70, llama=0.70, mistral=0.70),
        "action_direct": condition(0.72, llama=0.72, mistral=0.72),
        "action_empty": condition(0.73, llama=0.73, mistral=0.73),
        "action_reasoning": condition(0.74, llama=0.74, mistral=0.74),
    }

    selection = select_margin_arm(
        summaries,
        minimum_auroc_gain=0.05,
        maximum_source_auroc_loss=0.05,
    )

    assert selection["selected"] is None


def test_load_selected_rejects_failed_selection(tmp_path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text('{"selected": null}\n')

    try:
        load_selected(selection)
    except ValueError as error:
        assert "valid action margin arm" in str(error)
    else:
        raise AssertionError("failed selection should not load")


def test_left_truncation_matches_deployment_contract() -> None:
    assert left_truncate_prompt_ids([1, 2, 3, 4], 3) == [2, 3, 4]
    assert left_truncate_prompt_ids([1, 2], 3) == [1, 2]
