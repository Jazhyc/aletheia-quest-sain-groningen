from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]


def test_reasoning_intent_logits_contract() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "configs"),
    ):
        cfg = compose(
            config_name="judge_ensemble/blackbox_reasoning_intent_logits_v1"
        )

    assert cfg.judge.mode == "logits"
    assert cfg.judge.include_reasoning
    assert cfg.judge.reasoning_max_chars == 4000
    assert cfg.judge.reasoning_truncation == "tail"
    assert cfg.judge.logit_prefix == "Prediction:"
    assert [target.score for target in cfg.judge.logit_targets] == [0.0, 1.0]
    assert [list(target.texts) for target in cfg.judge.logit_targets] == [
        ["0"],
        ["1"],
    ]
    assert cfg.ensemble.order == "member"
    assert [member.name for member in cfg.ensemble.members] == [
        "literal_deceptive_intent",
        "resolved_intent",
    ]
