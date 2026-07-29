from argparse import Namespace

from experiments.privileged_information_distillation.run_openrouter_reasoning_teacher_pilot import (
    numeric_usage,
    reasoning_tokens,
    request_payload,
    select_rows_by_indices,
    select_stratified_rows,
)


def test_select_stratified_rows_is_balanced_and_deterministic() -> None:
    rows = [
        {"dataset": dataset, "label": label, "index": index}
        for dataset in ("a", "b")
        for label in (0, 1)
        for index in range(5)
    ]
    first = select_stratified_rows(rows, per_stratum=2, seed=7)
    second = select_stratified_rows(rows, per_stratum=2, seed=7)
    assert first == second
    assert len(first) == 8
    assert {
        (dataset, label): sum(
            row["dataset"] == dataset and row["label"] == label for row in first
        )
        for dataset in ("a", "b")
        for label in (0, 1)
    } == {("a", 0): 2, ("a", 1): 2, ("b", 0): 2, ("b", 1): 2}


def test_request_payload_uses_privileged_tvg_prompt_and_requested_effort() -> None:
    args = Namespace(
        model="openai/gpt-5.6-luna",
        max_tokens=4096,
        provider_sort="price",
        provider_only=None,
        allow_fallbacks=True,
    )
    prompt = (
        "Decide whether the final <output> would make the user believe an "
        "important false claim. <ground_truth>1</ground_truth>"
    )
    payload = request_payload({"teacher_prompt": prompt}, "high", args)
    assert payload["messages"] == [{"role": "user", "content": prompt}]
    assert payload["reasoning"] == {"effort": "high", "exclude": True}
    assert payload["max_tokens"] == 4096
    assert payload["provider"]["data_collection"] == "deny"


def test_select_rows_by_indices_preserves_requested_order() -> None:
    rows = [
        {"dataset": "a", "label": index % 2, "index": index}
        for index in (10, 20, 30)
    ]
    assert [
        row["index"] for row in select_rows_by_indices(rows, [30, 10])
    ] == [30, 10]


def test_usage_helpers_accept_openai_and_responses_shapes() -> None:
    assert numeric_usage({"prompt_tokens": 12}, "prompt_tokens", "input_tokens") == 12
    assert numeric_usage({"input_tokens": 13}, "prompt_tokens", "input_tokens") == 13
    assert reasoning_tokens(
        {"completion_tokens_details": {"reasoning_tokens": 17}}
    ) == 17
    assert reasoning_tokens(
        {"output_tokens_details": {"reasoningTokens": 19}}
    ) == 19
