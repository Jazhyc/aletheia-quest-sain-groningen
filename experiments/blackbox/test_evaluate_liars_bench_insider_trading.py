from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.blackbox.evaluate_liars_bench_insider_trading import (
    notebook_user_content,
    truncate_head_tail,
    truncate_tail,
)


def test_truncation_helpers_preserve_requested_length_and_tail() -> None:
    text = "a" * 100 + "TAIL"
    assert len(truncate_tail(text, 40)) == 40
    assert truncate_tail(text, 40).endswith("TAIL")
    assert len(truncate_head_tail(text, 40)) == 40
    assert truncate_head_tail(text, 40).endswith("TAIL")


def test_notebook_content_separately_preserves_final_output() -> None:
    messages = [
        {"role": "user", "content": "u" * 4000},
        {"role": "system", "content": "s" * 4000},
        {"role": "assistant", "content": "answer"},
    ]
    content, flags = notebook_user_content(messages)
    assert "<output>\nanswer\n</output>" in content
    assert flags["message_char_truncated"]
    assert flags["context_char_truncated"]
