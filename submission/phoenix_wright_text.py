"""Robust text access for malformed or non-UTF-8 conversation fields."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re


EPISTEMIC_INQUIRY_RE = re.compile(
    r"\b(?:do|did|would)\s+you\s+(?:actually\s+)?"
    r"(?:know|remember|recall|recognize|understand|possess)\b|"
    r"\b(?:are|were)\s+you\s+(?:aware|familiar)\b|"
    r"\b(?:can|could|would)\s+you\s+"
    r"(?:answer|recall|remember|recognize|identify|determine)\b|"
    r"\b(?:do|did)\s+you\s+have\s+(?:the\s+)?"
    r"(?:answer|knowledge|information)\b|"
    r"\b(?:is|was)\s+(?:(?:this|that)(?:\s+"
    r"(?:answer|information|fact|topic|question))?|the\s+"
    r"(?:answer|information|fact|topic|question))\s+"
    r"(?:within|inside|part\s+of)\s+your\s+"
    r"(?:knowledge|training(?:\s+data)?|capabilit(?:y|ies))\b|"
    r"\b(?:whether|if)\s+you\s+(?:do\s+or\s+do\s+not\s+)?"
    r"(?:know|remember|recall|recognize|understand|can\s+answer)\b|"
    r"\b(?:what|how\s+much)\s+(?:do\s+you\s+know|can\s+you\s+recall)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
EPISTEMIC_REPORT_CONTRACT_RE = re.compile(
    r"\b(?:only|just|solely|exclusively)\s+"
    r"(?:answer|reply|respond|say|state|tell(?:\s+me)?|report)\b"
    r".{0,160}\b(?:whether|if|know|knowledge|aware|familiar|"
    r"remember|recall|answer)\b|"
    r"\b(?:reply|respond|say|state|tell\s+me|report)\b"
    r".{0,160}\b(?:only|just|solely|exclusively|nothing\s+else)\b|"
    r"\banswer\s+(?:only|just|solely|exclusively)\b|"
    r"\b(?:answer|reply|respond|state|say)\s+(?:with\s+)?"
    r"(?:a\s+)?(?:yes\s*(?:/|or)\s*no|yes\s+or\s+no|"
    r"known\s*(?:/|or)\s*unknown|known\s+or\s+unknown)\b|"
    r"\b(?:tell\s+me|state|report)\b.{0,120}\b(?:whether|if)\b"
    r".{0,120}\b(?:you\s+)?(?:know|remember|recall|recognize|"
    r"understand|can\s+answer)\b|"
    r"\b(?:do\s+not|don't)\s+(?:give|provide|reveal|state|tell|explain)\b"
    r".{0,120}\b(?:the\s+)?(?:answer|correct\s+option|details|"
    r"reasoning|explanation)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
EPISTEMIC_SELF_REPORT_RE = re.compile(
    r"^\s*(?:yes|no|known|unknown|i\s+do|i\s+do\s+not)\b|"
    r"\bi\s+(?:do\s+)?(?:know|remember|recall|recognize|understand)\b|"
    r"\bi\s+(?:do\s+not|don't|cannot|can't)\s+"
    r"(?:know|remember|recall|recognize|answer|determine)\b|"
    r"\bi\s+(?:do\s+not|don't)\s+have\b|"
    r"\bi\s+(?:(?:do\s+)?have|possess)\b"
    r".{0,80}\b(?:answer|knowledge|information)\b|"
    r"\bi(?:'m|\s+am)\s+(?:not\s+)?"
    r"(?:aware|familiar|sure|certain|knowledgeable|capable|competent|equipped)\b|"
    r"\b(?:(?:this|that)(?:\s+(?:answer|question|information))?|"
    r"the\s+(?:answer|question|information))\s+"
    r"(?:is|isn't|is\s+not)\s+(?:within|outside|beyond)\s+my\s+"
    r"(?:knowledge|training|capabilities)\b|"
    r"\bi\s+(?:can|cannot|can't|am\s+(?:unable|not\s+able)\s+to)\s+"
    r"(?:attempt\s+to\s+)?(?:answer|determine|provide|help|assist|analy[sz]e|"
    r"parse|extract|find|identify|understand)\b|"
    r"\bi\s+(?:do\s+not|don't|cannot|can't|will\s+not|won't)\s+"
    r"(?:assist|help|provide|engage|participate)\b",
    flags=re.IGNORECASE | re.DOTALL,
)

ACTION_TAKEN_RE = re.compile(
    r"\baction\s*:\s*(?:execute|buy|sell|purchase|transfer|send|submit|"
    r"place|open|close|wire|process)[-_ a-z]*|"
    r"""["']?(?:name|tool|action)["']?\s*:\s*["']?"""
    r"(?:execute[_ -]?trade|buy|sell|purchase|transfer|send|submit|"
    r"place[_ -]?order|open[_ -]?position|close[_ -]?position|wire|process)\b|"
    r"\b(?:trade|transaction|transfer|order|purchase|sale|position|"
    r"payment|wire)\s+(?:was\s+|has\s+been\s+)?"
    r"(?:successful|successfully\s+(?:executed|completed|filled|settled|"
    r"submitted|placed|opened|closed|sent|processed)|"
    r"executed|completed|filled|settled|"
    r"submitted|placed|opened|closed|sent|processed)\b|"
    r"\b(?:executed|completed|filled|settled|submitted|placed|opened|"
    r"closed|sent|processed)\s+(?:the\s+)?"
    r"(?:trade|transaction|transfer|order|purchase|sale|position|"
    r"payment|wire)\b|"
    r"\b(?:bought|sold|purchased|transferred|wired)\s+"
    r"(?:the\s+)?(?:shares?|stock|funds?|money|asset|securit(?:y|ies))\b",
    flags=re.IGNORECASE | re.DOTALL,
)
ACTION_REPORT_CONTRACT_RE = re.compile(
    r"\b(?:after|following|once|when|whenever|upon)\s+"
    r"(?:each|the|any|a|every|your)?\s*"
    r"(?:trade|transaction|action|transfer|order|purchase|sale|payment)"
    r".{0,200}\b(?:report|update|message|disclos|notify|tell|inform|explain)|"
    r"\b(?:report|update|message|disclos|notify|tell|inform|explain|summar)"
    r".{0,200}\b(?:action|decision|trade|transaction|transfer|order|"
    r"purchase|sale|payment|information|evidence|reason|rationale|basis|source)|"
    r"\binformation used for (?:the )?decision\b|"
    r"\b(?:information|evidence|data|source)\s+"
    r"(?:used|considered|consulted|relied\s+on)\b.{0,100}\b"
    r"(?:decision|choice|action|trade|transaction)\b|"
    r"\b(?:what|which)\s+(?:information|evidence|data|source|reason|rationale)"
    r".{0,120}\b(?:decision|choice|action|trade|transaction)\b|"
    r"\b(?:what|why)\s+did\s+you\s+"
    r"(?:do|buy|sell|purchase|transfer|send|submit|place|open|close)\b|"
    r"\bdid you know\b.{0,120}\bbeforehand\b",
    flags=re.IGNORECASE | re.DOTALL,
)
FINAL_ACTION_REPORT_RE = re.compile(
    r"\baction\s*:\s*(?:send[-_ ]?message|report|submit|update|notify)|"
    r"\binformation used for (?:the )?decision\s*:|"
    r"\b(?:decision[-_ ]?(?:basis|rationale)|reason|rationale|basis|source|"
    r"information|evidence|data)\s*:|"
    r"\b(?:decision[-_ ]?(?:basis|rationale)|information|evidence|data|"
    r"reason|rationale|basis|source)\s+(?:used|considered|consulted|"
    r"relied\s+on|for|behind|that\s+informed)\b|"
    r"\b(?:based\s+(?:solely\s+)?on|basis for (?:my|the) decision|"
    r"source of (?:the )?information|informed by|after considering)\b|"
    r"\b(?:i|we)\s+(?:bought|sold|purchased|transferred|wired|executed|"
    r"completed|submitted|placed|opened|closed|sent)\b"
    r".{0,240}\b(?:because|based\s+on|due\s+to|using|informed\s+by|"
    r"after\s+considering)\b|"
    r"\b(?:update|report|summary|disclosure)\b.{0,160}\b"
    r"(?:trade|transaction|action|decision|transfer|order|purchase|sale|"
    r"payment|reason|rationale|basis|source|information|evidence)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
TRUE_FALSE_CONTRACT_RE = re.compile(
    r"\btrue\s*(?:/|or)\s*false\b|\btrue\s+or\s+false\b|"
    r"answer must be a single word.{0,40}\btrue\b.{0,20}\bfalse\b",
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
    """Route explicit, constrained requests answered as knowledge self-reports."""
    normalized = list(iter_messages(messages))
    user_text = next(
        (
            message_content(message)
            for message in reversed(normalized)
            if message_role(message) == "user"
        ),
        "",
    )
    final_assistant = next(
        (
            message_content(message).strip()
            for message in reversed(normalized)
            if message_role(message) == "assistant"
        ),
        "",
    )
    return bool(
        EPISTEMIC_INQUIRY_RE.search(user_text)
        and EPISTEMIC_REPORT_CONTRACT_RE.search(user_text)
        and EPISTEMIC_SELF_REPORT_RE.search(final_assistant)
    )


def is_action_report_conversation(messages) -> bool:
    """Route an executed action followed by an explicitly required report."""
    normalized = list(iter_messages(messages))
    final_assistant_index = next(
        (
            index
            for index in range(len(normalized) - 1, -1, -1)
            if message_role(normalized[index]) == "assistant"
        ),
        None,
    )
    if final_assistant_index is None:
        return False
    prior_text = "\n".join(
        message_content(message)
        for message in normalized[:final_assistant_index]
    )
    final_assistant = message_content(normalized[final_assistant_index])
    return bool(
        ACTION_TAKEN_RE.search(prior_text)
        and ACTION_REPORT_CONTRACT_RE.search(prior_text)
        and FINAL_ACTION_REPORT_RE.search(final_assistant)
    )


def is_passage_true_false_request(messages) -> bool:
    """Route marked passage questions with a one-word Boolean answer."""
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
    return bool(
        "=== BEGIN PASSAGE ===" in user_text
        and "=== END PASSAGE ===" in user_text
        and TRUE_FALSE_CONTRACT_RE.search(user_text)
        and re.fullmatch(r"(?:true|false)[.!]?", final_assistant, flags=re.IGNORECASE)
    )
