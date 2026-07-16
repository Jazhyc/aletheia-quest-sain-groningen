"""Label-blind routing for explicit action-report conversations."""

from __future__ import annotations

import re
from typing import Any

from submission.phoenix_wright_text import (
    iter_messages,
    message_content,
    message_role,
)


ACTION_TAKEN_RE = re.compile(
    r"\baction\s*:\s*(?:execute|buy|sell|transfer|send|submit|place)[-_ a-z]*|"
    r"\b(?:trade|transaction|transfer|order)\s+(?:was\s+)?(?:successful|executed|completed)",
    flags=re.IGNORECASE,
)
REPORT_CONTRACT_RE = re.compile(
    r"\b(?:after|following)\s+(?:each|the|any)\s+"
    r"(?:trade|transaction|action).{0,160}\b(?:report|update|message|disclos)|"
    r"\b(?:report|update|message|disclos).{0,160}"
    r"\b(?:action|decision|trade|transaction|information|basis|source)|"
    r"\binformation used for (?:the )?decision\b|"
    r"\bdid you know\b.{0,120}\bbeforehand\b",
    flags=re.IGNORECASE | re.DOTALL,
)
FINAL_REPORT_RE = re.compile(
    r"\baction\s*:\s*(?:send[-_ ]?message|report|submit)|"
    r"\binformation used for (?:the )?decision\s*:|"
    r"\b(?:based solely on|basis for (?:my|the) decision|source of (?:the )?information)\b",
    flags=re.IGNORECASE,
)


def is_action_report_conversation(messages: Any) -> bool:
    """Return whether an executed action is followed by an explicit report."""
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
        and REPORT_CONTRACT_RE.search(full_text)
        and FINAL_REPORT_RE.search(final_assistant)
    )
