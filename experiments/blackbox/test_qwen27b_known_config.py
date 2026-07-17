from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]


def compose_from(directory: str, name: str):
    with initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "configs" / directory).resolve()),
    ):
        return compose(config_name=name)


def test_qwen27b_single_known_member_matches_frozen_ensemble() -> None:
    ensemble = compose_from(
        "judge_ensemble",
        "blackbox_reasoning_qwen27b_ensemble_dks_member4096_v1",
    )
    single = compose_from(
        "single_judges", "blackbox_reasoning_qwen27b_known4096_v1"
    )

    assert single.judge.model == ensemble.judge.model == "Qwen/Qwen3.5-27B"
    assert single.judge.max_tokens == ensemble.judge.max_tokens == 4096
    assert single.judge.prompt == ensemble.ensemble.members[1].prompt
    assert single.scoring.threshold == ensemble.scoring.threshold == 0.01
