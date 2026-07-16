"""Robust text access for malformed or non-UTF-8 conversation fields."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


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
