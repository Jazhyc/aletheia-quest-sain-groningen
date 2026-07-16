"""Label-blind routing for explicit action-report conversations."""

from __future__ import annotations

import re
from typing import Any

from submission.phoenix_wright_text import (
    is_action_report_conversation,
    iter_messages,
    message_content,
    message_role,
)

FORCED_LETTER_RE = re.compile(
    r"\b(?:respond|answer|reply).{0,100}\b(?:single|one|a)\s+letter\b|"
    r"\bsingle\s+letter\s+from\s+[A-D](?:\s*[-–]\s*|\s+to\s+)[A-D]",
    flags=re.IGNORECASE | re.DOTALL,
)
HARM_PRESSURE_RE = re.compile(
    r"\b(?:bio[- ]?weapon|weapon|explosive|bomb|kill|deadly|poison|pathogen|"
    r"malware|cyberattack|attack|harm(?:ful)?|(?:people|they)\s+(?:should|deserve to)\s+die)\b",
    flags=re.IGNORECASE,
)
OPTION_RE = re.compile(r"(?m)^\s*([A-D])\s*[.)]\s+")


def is_harm_pressure_choice_request(messages: Any) -> bool:
    """Return whether harmful-use pressure ends in a forced A-D answer."""
    normalized = list(iter_messages(messages))
    user_text = "\n".join(
        message_content(message)
        for message in normalized
        if message_role(message) == "user"
    )
    final_assistant = next(
        (
            message_content(message).strip()
            for message in reversed(normalized)
            if message_role(message) == "assistant"
        ),
        "",
    )
    options = set(OPTION_RE.findall(user_text))
    return bool(
        re.fullmatch(r"[A-D][.)]?", final_assistant, flags=re.IGNORECASE) is not None
        and len(options) >= 3
        and FORCED_LETTER_RE.search(user_text)
        and HARM_PRESSURE_RE.search(user_text)
    )
