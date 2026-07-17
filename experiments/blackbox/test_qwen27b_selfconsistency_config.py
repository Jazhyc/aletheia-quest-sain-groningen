from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]


def compose_config(directory: str, name: str):
    with initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "configs" / directory).resolve()),
    ):
        return compose(config_name=name)


def test_self_consistency_repeats_exact_known_prompt_at_matched_cost() -> None:
    single = compose_config(
        "single_judges", "blackbox_reasoning_qwen27b_known4096_v1"
    )
    repeated = compose_config(
        "judge_ensemble",
        "blackbox_reasoning_qwen27b_known_selfconsistency3_t06_v1",
    )

    assert repeated.judge.model == single.judge.model == "Qwen/Qwen3.5-27B"
    assert repeated.judge.max_tokens == single.judge.max_tokens == 4096
    assert repeated.judge.temperature == 0.6
    assert repeated.ensemble.aggregation == "median"
    assert len(repeated.ensemble.members) == 3
    assert all(
        member.prompt == single.judge.prompt for member in repeated.ensemble.members
    )
