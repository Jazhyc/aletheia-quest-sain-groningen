from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.blackbox.evaluate_liars_bench_hpkr import parse_prediction
from experiments.liars_bench_hpkr_prompts.evaluate import PROMPTS, render_prompts
from experiments.liars_bench_hpkr_prompts.evaluate_short import PROMPTS as SHORT_PROMPTS


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


def test_prompts_require_binary_output_without_revealing_answer() -> None:
    assert len(PROMPTS) == 5
    assert all("Prediction:<0 or 1>" in prompt for prompt in PROMPTS.values())
    assert all("do not reveal operational details" in prompt.lower() for prompt in PROMPTS.values())


def test_render_and_parse_prompt() -> None:
    rendered = render_prompts(
        FakeTokenizer(),
        __import__("pandas").DataFrame({"messages": [[{"role": "assistant", "content": "I do not know."}]]}),
        PROMPTS["alignment_aware"],
    )
    assert "<output>" in rendered[0]
    assert parse_prediction("Prediction:1") == 1


def test_short_prompts_have_no_reasoning_field() -> None:
    assert len(SHORT_PROMPTS) == 2
    assert all(prompt.endswith("Output only: Prediction:<0 or 1>") for prompt in SHORT_PROMPTS.values())
