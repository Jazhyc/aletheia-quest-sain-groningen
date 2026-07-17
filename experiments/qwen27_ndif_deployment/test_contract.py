from pathlib import Path

from hydra import compose, initialize_config_dir

from experiments.blackbox.run_judge import build_prompt, parse_rating as local_parse_rating
from experiments.qwen27_ndif_deployment.contract import (
    MAX_NEW_TOKENS,
    MAX_PROMPT_CHARS,
    MODEL,
    STOP_STRINGS,
    aggregate_member_ratings,
    member_major_prompts,
    parse_rating,
    render_prompt,
)


ROOT = Path(__file__).resolve().parents[2]


def frozen_config():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "configs" / "judge_ensemble"),
    ):
        return compose(
            config_name="blackbox_reasoning_qwen27b_ensemble_dks_member4096_v1"
        )


def test_constants_match_frozen_qwen27_config() -> None:
    config = frozen_config()
    assert MODEL == config.judge.model
    assert MAX_PROMPT_CHARS == config.judge.max_prompt_chars
    assert MAX_NEW_TOKENS == config.judge.max_tokens
    assert STOP_STRINGS == tuple(f"Rating: {rating}" for rating in range(1, 8))


def test_independent_renderer_matches_local_evaluator() -> None:
    config = frozen_config()
    member_prompt = str(config.ensemble.members[0].prompt)
    messages = [
        {"role": "system", "content": "Be accurate."},
        {"role": "user", "content": "Is A true?"},
        {"role": "assistant", "content": "No."},
    ]
    assert render_prompt(messages, member_prompt) == build_prompt(
        messages,
        member_prompt,
        MAX_PROMPT_CHARS,
        append_prefix=None,
        context_truncation="tail",
    )


def test_member_major_order_and_aggregation_are_frozen() -> None:
    rendered = member_major_prompts(
        [[{"role": "assistant", "content": "A"}], [{"role": "assistant", "content": "B"}]],
        ["p0", "p1", "p2"],
    )
    assert [(member, row) for member, row, _ in rendered] == [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)
    ]
    assert aggregate_member_ratings([1, None, 1]) == (False, 0.0)
    deceptive, score = aggregate_member_ratings([1, 2, None])
    assert deceptive is True
    assert score == 1 / 6


def test_rating_parser_matches_local_contract() -> None:
    samples = (
        "analysis\nRating: 1",
        "Rating: 2\nrevision\nFinal rating: 7",
        "assistantfinalRating: 6",
        "4/7",
        "no verdict",
    )
    for sample in samples:
        assert parse_rating(sample) == local_parse_rating(
            sample, rating_min=1, rating_max=7
        )
