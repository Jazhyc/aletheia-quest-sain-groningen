import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))

from phoenix_wright_text import (
    is_action_report_conversation,
    is_epistemic_report_request,
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
