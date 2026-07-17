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


def test_qwen27b_nothink_matches_regular_qwen9b_condition() -> None:
    qwen9b = compose_from(
        "single_judges", "blackbox_reasoning_nothink_t512_v1"
    )
    qwen27b = compose_from(
        "single_judges", "blackbox_reasoning_qwen27b_nothink_t512_v1"
    )

    assert qwen27b.judge.model == "Qwen/Qwen3.5-27B"
    assert qwen27b.judge.prompt == qwen9b.judge.prompt
    assert qwen27b.judge.use_chat_template is True
    assert qwen27b.judge.enable_thinking is False
    assert qwen27b.judge.max_tokens == qwen9b.judge.max_tokens == 512
    assert qwen27b.scoring.threshold == qwen9b.scoring.threshold == 0.5
