"""Shared prompt and parsing helpers for privileged-information distillation."""

from __future__ import annotations

import re
from typing import Any


SUMMARY_PATTERN = re.compile(
    r"<reasoning_summary>\s*(.*?)\s*</reasoning_summary>\s*Prediction:\s*([01])\b",
    flags=re.DOTALL,
)
SUMMARY_ONLY_PATTERN = re.compile(
    r"<reasoning_summary>\s*(.*?)\s*</reasoning_summary>",
    flags=re.DOTALL,
)


def safe_text(value: Any) -> str:
    return str(value if value is not None else "").encode(
        "utf-8", errors="replace"
    ).decode("utf-8", errors="replace")


def merge_messages(messages: Any) -> list[dict[str, str]]:
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        messages = [{"role": "unknown", "content": messages}]
    merged: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            item = {"role": "unknown", "content": item}
        role = safe_text(item.get("role", "unknown"))
        content = safe_text(item.get("content", ""))
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + content
        else:
            merged.append({"role": role, "content": content})
    return merged


def truncate_text(text: str, max_chars: int, mode: str) -> str:
    if len(text) <= max_chars:
        return text
    if mode == "tail":
        return text[-max_chars:]
    if mode == "head":
        return text[:max_chars]
    if mode == "head_tail":
        marker = "\n\n[... middle truncated ...]\n\n"
        available = max(0, max_chars - len(marker))
        head = available // 2
        return text[:head] + marker + text[-(available - head):]
    raise ValueError(f"unknown truncation mode: {mode}")


def format_example(messages: Any, max_chars: int, truncation: str) -> tuple[str, str]:
    merged = merge_messages(messages)
    output = next(
        (message["content"] for message in reversed(merged) if message["role"] == "assistant"),
        "",
    )
    context = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in merged
    )
    return truncate_text(context, max_chars, truncation), output


def build_student_prompt(
    messages: Any,
    prompt_template: str,
    max_chars: int,
    truncation: str,
) -> str:
    context, output = format_example(messages, max_chars, truncation)
    return f"{prompt_template}\n\n<context>\n{context}\n</context>\n\n<output>\n{output}\n</output>"


def build_teacher_prompt(student_prompt: str, teacher_template: str, label: int) -> str:
    privileged = teacher_template.replace("__GROUND_TRUTH__", str(int(label)))
    _, separator, evidence = student_prompt.partition("<context>")
    if not separator:
        raise ValueError("student prompt is missing <context>")
    return f"{privileged}\n\n<context>{evidence}"


def extract_harmony_final(raw_completion: str) -> str:
    """Return only GPT-OSS's visible final channel, excluding innate analysis."""
    text = safe_text(raw_completion)
    markers = ("assistantfinal", "<|channel|>final<|message|>")
    positions = [(text.rfind(marker), marker) for marker in markers]
    position, marker = max(positions, key=lambda item: item[0])
    return text[position + len(marker):].strip() if position >= 0 else text.strip()


def parse_teacher_target(
    raw_completion: str,
    expected_prediction: int | None = None,
) -> tuple[str, int] | None:
    final = extract_harmony_final(raw_completion)
    matches = list(SUMMARY_PATTERN.finditer(final))
    if matches:
        summary, prediction = matches[-1].groups()
        prediction = int(prediction)
    elif expected_prediction in (0, 1):
        summary_matches = list(SUMMARY_ONLY_PATTERN.finditer(final))
        if not summary_matches:
            return None
        summary = summary_matches[-1].group(1)
        prediction = int(expected_prediction)
    else:
        return None
    summary = " ".join(summary.split())
    if not summary:
        return None
    return summary, prediction


def format_student_target(summary: str, prediction: int) -> str:
    return (
        f"<reasoning_summary>\n{summary.strip()}\n</reasoning_summary>\n"
        f"Prediction:{int(prediction)}"
    )
