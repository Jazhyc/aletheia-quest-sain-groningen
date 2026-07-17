from experiments.blackbox.run_judge import generate_with_optional_batches


class FakeLlm:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompts, sampling, **kwargs):
        self.calls.append((prompts, sampling, kwargs))
        return prompts


def test_optional_batching_forwards_lora_request() -> None:
    llm = FakeLlm()
    request = object()

    outputs = generate_with_optional_batches(
        llm,
        ["a", "b", "c"],
        "sampling",
        2,
        lora_request=request,
    )

    assert outputs == ["a", "b", "c"]
    assert [call[0] for call in llm.calls] == [["a", "b"], ["c"]]
    assert all(call[2]["lora_request"] is request for call in llm.calls)
