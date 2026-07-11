import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.blackbox.run_judge import parse_rating


def test_parse_rating_accepts_gpt_oss_harmony_final_channel() -> None:
    reply = "analysisThe output is false.assistantfinalRating: 7"

    assert parse_rating(reply, rating_min=1, rating_max=7) == 7


def test_parse_rating_does_not_treat_analysis_rating_as_final_without_prefix() -> None:
    reply = "The possible rating is unclear, perhaps between 1 and 7."

    assert parse_rating(reply, rating_min=1, rating_max=7) is None
