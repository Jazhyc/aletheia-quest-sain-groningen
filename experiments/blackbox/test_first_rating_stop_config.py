from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]


def test_qwen27_first_rating_stop_preserves_exact_verdict_strings() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "configs" / "judge_ensemble"),
    ):
        config = compose(
            config_name="blackbox_reasoning_qwen27b_ensemble_dks_first_rating_stop_v1"
        )

    assert config.judge.include_stop_str_in_output is True
    assert list(config.judge.stop_strings) == [f"Rating: {rating}" for rating in range(1, 8)]
    assert config.judge.max_tokens == 4096
    assert config.ensemble.order == "member"


def test_qwen9_first_rating_stop_uses_same_frozen_contract() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "configs" / "judge_ensemble"),
    ):
        config = compose(
            config_name="blackbox_reasoning_qwen9b_ensemble_dks_first_rating_stop_v1"
        )

    assert config.judge.model == "Qwen/Qwen3.5-9B"
    assert config.judge.include_stop_str_in_output is True
    assert list(config.judge.stop_strings) == [f"Rating: {rating}" for rating in range(1, 8)]
    assert config.judge.max_tokens == 4096
