"""Robust text access for malformed or non-UTF-8 conversation fields."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re


EPISTEMIC_REPORT_REQUEST_RE = re.compile(
    r"(?:do you know|whether you (?:do or do not )?know|know the answer|"
    r"have (?:the )?(?:answer|knowledge)|are you (?:aware|familiar)|"
    r"can you (?:recall|remember)).{0,240}(?:only|just|nothing else)|"
    r"(?:only|just).{0,240}(?:whether|if).{0,120}(?:you know|know the answer)",
    flags=re.IGNORECASE | re.DOTALL,
)


def safe_text(value) -> str:
    """Return text that can be passed to Rust-backed Hugging Face tokenizers."""
    return str(value).encode("utf-8", "replace").decode("utf-8")


def iter_messages(messages) -> list:
    """Return a list-like dialogue even if a row has malformed message structure."""
    if messages is None:
        return []
    if isinstance(messages, Mapping):
        return [messages]
    if isinstance(messages, (str, bytes)):
        return [{"role": "", "content": messages}]
    if isinstance(messages, Iterable):
        return list(messages)
    return [{"role": "", "content": messages}]


def message_field(message, field: str, default="") -> str:
    """Safely read one message field, treating non-mapping messages as content."""
    if isinstance(message, Mapping):
        value = message.get(field, default)
        return safe_text(default if value is None else value)
    if field == "content":
        return safe_text(message)
    return safe_text(default)


def message_role(message) -> str:
    return message_field(message, "role")


def message_content(message) -> str:
    return message_field(message, "content")


def is_epistemic_report_request(messages) -> bool:
    """Route only explicit requests for a constrained knowledge self-report."""
    user_text = "\n".join(
        message_content(message)
        for message in iter_messages(messages)
        if message_role(message) == "user"
    )
    return bool(EPISTEMIC_REPORT_REQUEST_RE.search(user_text))
