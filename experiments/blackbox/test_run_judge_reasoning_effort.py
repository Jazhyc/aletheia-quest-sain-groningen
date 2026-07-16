from experiments.blackbox.run_judge import render_chat_prompts


def test_render_chat_prompts_passes_harmony_reasoning_effort() -> None:
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return f"rendered-{kwargs['reasoning_effort']}"

    rendered = render_chat_prompts(
        Tokenizer(),
        ["first", "second"],
        enable_thinking=False,
        reasoning_effort="high",
    )

    assert rendered == ["rendered-high", "rendered-high"]
    assert [call[0][0]["content"] for call in calls] == ["first", "second"]
    assert all(call[1]["enable_thinking"] is False for call in calls)
