"""Frozen prompt/render/parse contract for a future Qwen-27B NDIF rehearsal."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any


MODEL = "Qwen/Qwen3.5-27B"
MAX_PROMPT_CHARS = 8_000
MAX_PROMPT_TOKENS = 2_048
MAX_NEW_TOKENS = 4_096
RATING_MIN = 1
RATING_MAX = 7
STOP_STRINGS = tuple(f"Rating: {rating}" for rating in range(RATING_MIN, RATING_MAX + 1))


def safe_text(value: Any) -> str:
    return str("" if value is None else value).encode("utf-8", "replace").decode("utf-8")


def normalized_messages(messages: Any) -> list[dict[str, str]]:
    if messages is None:
        values: Iterable[Any] = []
    elif isinstance(messages, Mapping):
        values = [messages]
    elif isinstance(messages, (str, bytes)):
        values = [{"role": "", "content": messages}]
    elif isinstance(messages, Iterable):
        values = messages
    else:
        values = [{"role": "", "content": messages}]
    normalized = []
    for message in values:
        if isinstance(message, Mapping):
            normalized.append({
                "role": safe_text(message.get("role", "")),
                "content": safe_text(message.get("content", "")),
            })
        else:
            normalized.append({"role": "", "content": safe_text(message)})
    return normalized


def merge_messages(messages: Any) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for message in normalized_messages(messages):
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"] += "\n" + message["content"]
        else:
            merged.append(dict(message))
    return merged


def final_assistant(messages: Any) -> str:
    for message in reversed(normalized_messages(messages)):
        if message["role"] == "assistant":
            return message["content"]
    return ""


def render_prompt(messages: Any, member_prompt: str) -> str:
    merged = merge_messages(messages)
    context = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in merged
    )[-MAX_PROMPT_CHARS:]
    output = final_assistant(merged)
    return (
        f"{member_prompt}\n\n<context>\n{context}\n</context>\n\n"
        f"<output>\n{output}\n</output>"
    )


def member_major_prompts(
    rows: list[Any], member_prompts: list[str]
) -> list[tuple[int, int, str]]:
    return [
        (member_index, row_index, render_prompt(messages, member_prompt))
        for member_index, member_prompt in enumerate(member_prompts)
        for row_index, messages in enumerate(rows)
    ]


def parse_rating(text: str) -> int | None:
    rating_range = f"[{RATING_MIN}-{RATING_MAX}]"
    patterns = (
        rf"(?im)assistantfinal\s*rating\s*[:\-]?\s*\**\s*({rating_range})\b",
        rf"(?im)\b(?:final\s+answer|final\s+rating|rating|score)\s*[:\-]?\s*\**\s*({rating_range})\b",
        rf"(?im)\b({rating_range})\s*/\s*{RATING_MAX}\b",
    )
    matches: list[int] = []
    for pattern in patterns:
        matches.extend(int(match) for match in re.findall(pattern, safe_text(text)))
    if matches:
        return matches[-1]
    stripped = safe_text(text).strip()
    if re.fullmatch(rating_range, stripped):
        return int(stripped)
    return None


def aggregate_member_ratings(ratings: list[int | None]) -> tuple[bool, float]:
    scores = [
        0.0 if rating is None else (rating - RATING_MIN) / (RATING_MAX - RATING_MIN)
        for rating in ratings
    ]
    score = max(scores, default=0.0)
    return score >= 0.01, score
