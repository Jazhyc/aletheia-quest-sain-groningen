from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]


def compose_config(directory: str, name: str):
    with initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "configs" / directory).resolve()),
    ):
        return compose(config_name=name)


def test_sweep_is_auroc_first_continuous_and_preserves_control_prompt():
    sweep = compose_config(
        "judge_ensemble",
        "blackbox_persona_belief_auroc_sweep_v1",
    )
    control = compose_config(
        "single_judges",
        "blackbox_reasoning_nothink_truth_value_v1",
    )

    assert sweep.judge.mode == "logits"
    assert sweep.judge.include_reasoning is False
    assert sweep.ensemble.order == "member"
    assert [member.name for member in sweep.ensemble.members] == [
        "truth_value_guard_control",
        "underlying_assistant_general",
        "underlying_assistant_epistemic",
    ]
    assert sweep.ensemble.members[0].prompt == control.judge.prompt


def test_prompt_variants_keep_persona_instruction_insufficient():
    sweep = compose_config(
        "judge_ensemble",
        "blackbox_persona_belief_auroc_sweep_v1",
    )
    for member in sweep.ensemble.members[1:]:
        prompt = member.prompt.lower()
        assert "underlying assistant" in prompt
        assert "alone is insufficient" in prompt
        assert "fictional" in prompt
