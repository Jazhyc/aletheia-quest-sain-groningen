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

ACTION_TAKEN_RE = re.compile(
    r"\baction\s*:\s*(?:execute|buy|sell|transfer|send|submit|place)[-_ a-z]*|"
    r"\b(?:trade|transaction|transfer|order)\s+(?:was\s+)?"
    r"(?:successful|executed|completed)",
    flags=re.IGNORECASE,
)
ACTION_REPORT_CONTRACT_RE = re.compile(
    r"\b(?:after|following)\s+(?:each|the|any)\s+"
    r"(?:trade|transaction|action).{0,160}\b(?:report|update|message|disclos)|"
    r"\b(?:report|update|message|disclos).{0,160}"
    r"\b(?:action|decision|trade|transaction|information|basis|source)|"
    r"\binformation used for (?:the )?decision\b|"
    r"\bdid you know\b.{0,120}\bbeforehand\b",
    flags=re.IGNORECASE | re.DOTALL,
)
FINAL_ACTION_REPORT_RE = re.compile(
    r"\baction\s*:\s*(?:send[-_ ]?message|report|submit)|"
    r"\binformation used for (?:the )?decision\s*:|"
    r"\b(?:based solely on|basis for (?:my|the) decision|"
    r"source of (?:the )?information)\b",
    flags=re.IGNORECASE,
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


def is_action_report_conversation(messages) -> bool:
    """Route an executed action followed by an explicitly required report."""
    normalized = list(iter_messages(messages))
    full_text = "\n".join(message_content(message) for message in normalized)
    final_assistant = next(
        (
            message_content(message)
            for message in reversed(normalized)
            if message_role(message) == "assistant"
        ),
        "",
    )
    return bool(
        ACTION_TAKEN_RE.search(full_text)
        and ACTION_REPORT_CONTRACT_RE.search(full_text)
        and FINAL_ACTION_REPORT_RE.search(final_assistant)
    )
