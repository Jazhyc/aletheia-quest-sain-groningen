import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.blackbox.run_judge import build_prompt, load_retrieval_cache, parse_rating


def test_parse_rating_accepts_gpt_oss_harmony_final_channel() -> None:
    reply = "analysisThe output is false.assistantfinalRating: 7"

    assert parse_rating(reply, rating_min=1, rating_max=7) == 7


def test_parse_rating_does_not_treat_analysis_rating_as_final_without_prefix() -> None:
    reply = "The possible rating is unclear, perhaps between 1 and 7."

    assert parse_rating(reply, rating_min=1, rating_max=7) is None


def test_build_prompt_adds_guarded_reference_material() -> None:
    prompt = build_prompt(
        [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}],
        "Judge this response.",
        1000,
        append_prefix=None,
        reference_material="- Example: Retrieved fact.",
    )

    assert "<reference_material>" in prompt
    assert "may be incomplete or irrelevant" in prompt
    assert "- Example: Retrieved fact." in prompt
    assert prompt.index("</output>") < prompt.index("<reference_material>")


def test_load_retrieval_cache_formats_passages(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text(
        '{"dataset":"demo","index":4,"passages":['
        '{"title":"Alpha","text":"First fact."},'
        '{"title":"Empty","text":""}]}\n'
    )

    assert load_retrieval_cache(path) == {("demo", 4): "- Alpha: First fact."}
