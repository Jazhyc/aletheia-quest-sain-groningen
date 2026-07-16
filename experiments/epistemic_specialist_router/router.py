"""High-precision routing for explicit epistemic self-report requests."""

from __future__ import annotations

import re
from typing import Any

from submission.phoenix_wright_text import (
    iter_messages,
    message_content,
    message_role,
)


EPISTEMIC_REPORT_REQUEST_RE = re.compile(
    r"(?:do you know|whether you (?:do or do not )?know|know the answer|"
    r"have (?:the )?(?:answer|knowledge)|are you (?:aware|familiar)|"
    r"can you (?:recall|remember)).{0,240}(?:only|just|nothing else)|"
    r"(?:only|just).{0,240}(?:whether|if).{0,120}(?:you know|know the answer)",
    flags=re.IGNORECASE | re.DOTALL,
)


def is_epistemic_report_request(messages: Any) -> bool:
    """Return whether the user explicitly requests only a knowledge self-report."""
    user_text = "\n".join(
        message_content(message)
        for message in iter_messages(messages)
        if message_role(message) == "user"
    )
    return bool(EPISTEMIC_REPORT_REQUEST_RE.search(user_text))
