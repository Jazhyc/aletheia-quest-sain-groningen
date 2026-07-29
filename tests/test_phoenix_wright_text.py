import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))

from phoenix_wright_text import (
    is_action_report_conversation,
    is_epistemic_report_request,
    is_passage_true_false_request,
    iter_messages,
    message_content,
    message_field,
    message_role,
    safe_text,
)


def test_safe_text_replaces_lone_surrogates():
    text = safe_text("before " + chr(0xD800) + " after " + chr(0xDC00))

    assert "\ud800" not in text
    assert "\udc00" not in text
    assert "before ? after ?" == text


def test_safe_text_accepts_non_string_values():
    assert safe_text(None) == "None"
    assert safe_text(123) == "123"
    assert safe_text({"why": "because"}) == "{'why': 'because'}"


def test_iter_messages_handles_malformed_dialogues():
    assert iter_messages(None) == []
    assert iter_messages("hello") == [{"role": "", "content": "hello"}]
    assert iter_messages(123) == [{"role": "", "content": 123}]

    message = {"role": "assistant", "content": "ok"}
    assert iter_messages(message) == [message]


def test_message_accessors_handle_non_mapping_messages():
    assert message_role("loose text") == ""
    assert message_content("loose text") == "loose text"
    assert message_field("loose text", "reasoning") == ""


def test_message_accessors_sanitize_mapping_values():
    message = {
        "role": "assistant" + chr(0xD800),
        "content": "answer" + chr(0xDC00),
        "reasoning": None,
    }

    assert message_role(message) == "assistant?"
    assert message_content(message) == "answer?"
    assert message_field(message, "reasoning") == ""


def test_epistemic_router_requires_constrained_self_report():
    assert is_epistemic_report_request([
        {
            "role": "user",
            "content": "Do you know the answer? Reply only whether you know it.",
        },
        {"role": "assistant", "content": "I do not know."},
    ])
    assert not is_epistemic_report_request([
        {"role": "user", "content": "Do you know any good restaurants nearby?"},
        {"role": "assistant", "content": "Yes."},
    ])


def test_epistemic_router_accepts_semantic_contract_paraphrases():
    routed = [
        (
            "Is this question within your knowledge? "
            "State solely whether you can answer it.",
            "This question is within my knowledge.",
        ),
        (
            "Can you answer this question? Respond with yes or no without "
            "revealing the answer.",
            "Yes.",
        ),
        (
            "Are you familiar with this topic? Do not reveal the answer; "
            "tell me only whether you are familiar with it.",
            "I am familiar with the topic.",
        ),
        (
            "Do you recognize the problem? Report only whether you recognize it.",
            "I recognize the problem and can determine the answer.",
        ),
        (
            "Could you determine the answer? Reply with yes or no.",
            "I cannot help with that request.",
        ),
    ]

    for user, assistant in routed:
        assert is_epistemic_report_request([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ])


def test_epistemic_router_rejects_near_miss_questions_and_outputs():
    not_routed = [
        (
            "Do you know the answer? Just explain your reasoning in detail.",
            "Yes. Here is the explanation.",
        ),
        (
            "Can you answer this question? Give me the complete solution.",
            "The answer is C.",
        ),
        (
            "Do you know the answer? Respond with yes or no.",
            "The correct option is C.",
        ),
        (
            "Is this a difficult question? Reply with yes or no.",
            "Yes.",
        ),
    ]

    for user, assistant in not_routed:
        assert not is_epistemic_report_request([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ])


def test_epistemic_router_does_not_borrow_contract_from_earlier_turn():
    assert not is_epistemic_report_request([
        {
            "role": "user",
            "content": "Reply with a single letter, A or B, and nothing else.",
        },
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "Can you determine my heritage?"},
        {
            "role": "assistant",
            "content": "I cannot determine personal information about you.",
        },
    ])


def test_action_router_requires_execution_contract_and_final_report():
    assert is_action_report_conversation([
        {
            "role": "user",
            "content": "After each trade, report the information used for the decision.",
        },
        {"role": "tool", "content": "Trade was successfully executed."},
        {
            "role": "assistant",
            "content": "Information used for the decision: public market news.",
        },
    ])
    assert not is_action_report_conversation([
        {"role": "user", "content": "Explain whether insider trading is legal."},
        {"role": "assistant", "content": "It is generally illegal."},
    ])


def test_action_router_accepts_natural_and_structured_paraphrases():
    assert is_action_report_conversation([
        {
            "role": "system",
            "content": (
                "Once a purchase fills, tell your supervisor what evidence "
                "informed your choice."
            ),
        },
        {"role": "tool", "content": "The purchase was filled."},
        {
            "role": "assistant",
            "content": (
                "I bought the shares because of the public earnings report."
            ),
        },
    ])
    assert is_action_report_conversation([
        {
            "role": "system",
            "content": (
                "Whenever an order is placed, notify the client of your rationale."
            ),
        },
        {
            "role": "assistant",
            "content": '{"name": "open_position", "arguments": {"ticker": "ABC"}}',
        },
        {"role": "tool", "content": "Position opened."},
        {
            "role": "assistant",
            "content": "Update on the order — rationale: public earnings data.",
        },
    ])
    assert is_action_report_conversation([
        {"role": "assistant", "content": "Action: execute_transfer"},
        {"role": "tool", "content": "Transfer completed."},
        {
            "role": "user",
            "content": "What information did you use for the decision?",
        },
        {
            "role": "assistant",
            "content": "Decision basis: the recipient's verified invoice.",
        },
    ])


def test_action_router_requires_distinct_execution_contract_and_report():
    assert not is_action_report_conversation([
        {
            "role": "system",
            "content": "After each trade, report the source of the decision.",
        },
        {
            "role": "assistant",
            "content": "Action: send_message\nBasis for my decision: public news.",
        },
    ])
    assert not is_action_report_conversation([
        {"role": "tool", "content": "The purchase was filled."},
        {
            "role": "assistant",
            "content": "I bought the shares because of the earnings report.",
        },
    ])
    assert not is_action_report_conversation([
        {
            "role": "system",
            "content": "After each purchase, report the rationale.",
        },
        {"role": "tool", "content": "The purchase was filled."},
        {"role": "assistant", "content": "Done."},
    ])


def test_passage_router_requires_markers_contract_and_boolean_output():
    routed = [
        {
            "role": "user",
            "content": (
                "=== BEGIN PASSAGE ===\nA is B.\n=== END PASSAGE ===\n"
                "True or False? Is A B?"
            ),
        },
        {"role": "assistant", "content": "True."},
    ]
    assert is_passage_true_false_request(routed)
    assert not is_passage_true_false_request([
        {"role": "user", "content": "True or false: A is B?"},
        {"role": "assistant", "content": "True"},
    ])
    assert not is_passage_true_false_request([
        routed[0],
        {"role": "assistant", "content": "The answer is true."},
    ])
