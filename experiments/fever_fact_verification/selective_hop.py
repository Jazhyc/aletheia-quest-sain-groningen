"""Shared contracts for label-blind selective evidence auditing."""

from __future__ import annotations

import re
from typing import Any


RELATIONS = {
    "DECISIVE_SUPPORT",
    "DECISIVE_CONTRADICTION",
    "TOPICAL",
    "INSUFFICIENT",
}
DECISIVE = {"DECISIVE_SUPPORT", "DECISIVE_CONTRADICTION"}
CANDIDATE_RE = re.compile(
    r"(?im)^\s*CANDIDATE_(\d+)\s*:\s*"
    r"(DECISIVE_SUPPORT|DECISIVE_CONTRADICTION|TOPICAL|INSUFFICIENT)\s*$"
)
SELECTED_RE = re.compile(r"(?im)^\s*SELECTED_ID\s*:\s*(\d+|NONE)\s*$")
DECISION_RE = re.compile(
    r"(?im)^\s*DECISION\s*:\s*"
    r"(DECISIVE_SUPPORT|DECISIVE_CONTRADICTION|ABSTAIN)\s*$"
)
QUERY_RE = re.compile(r"(?im)^\s*QUERY\s*:\s*(.*?)\s*$")
RATIONALE_RE = re.compile(r"(?im)^\s*RATIONALE\s*:\s*(.*?)\s*$")


def claim_key(row: dict[str, Any]) -> tuple[str, Any, int]:
    return str(row["dataset"]), row["index"], int(row["claim_index"])


def bounded_candidates(
    row: dict[str, Any], *, source_field: str, limit: int, max_chars: int
) -> list[dict[str, Any]]:
    """Return bounded, non-empty candidates without changing their rank."""
    output = []
    for candidate in row.get(source_field) or []:
        text = " ".join(str(candidate.get("text") or "").split())
        if not text:
            continue
        if len(text) > max_chars:
            text = text[: max_chars - 4].rstrip() + " ..."
        output.append({
            "title": " ".join(str(candidate.get("title") or "Wikipedia").split()),
            "text": text,
            "url": str(candidate.get("url") or ""),
            "page_rank": candidate.get("page_rank"),
            "sentence_index": candidate.get("sentence_index"),
        })
        if len(output) >= limit:
            break
    return output


def build_audit_prompt(proposition: str, candidates: list[dict[str, Any]]) -> str:
    rendered = "\n\n".join(
        f"CANDIDATE {position}\nTITLE: {candidate['title']}\nTEXT: {candidate['text']}"
        for position, candidate in enumerate(candidates, start=1)
    ) or "NO CANDIDATES"
    candidate_lines = "\n".join(
        f"CANDIDATE_{position}: <relation>"
        for position in range(1, len(candidates) + 1)
    )
    return f"""You are a label-blind evidence auditor. You are not deciding whether an
answer is deceptive, and you do not know any ground-truth label. Assess only
whether each candidate sentence directly settles the exact proposition below.

Relations:
- DECISIVE_SUPPORT: the sentence, with its title, directly establishes the
  proposition for the same entity, time, quantity, and polarity.
- DECISIVE_CONTRADICTION: the sentence directly establishes an incompatible
  fact for the same entity, time, quantity, and polarity.
- TOPICAL: it concerns the same broad subject but does not settle the
  proposition.
- INSUFFICIENT: it is ambiguous, concerns a different entity or time, or cannot
  establish either support or contradiction.

Be conservative. Shared words, a relevant page title, omitted information, or
your own background knowledge are not decisive evidence. If no candidate is
decisive, abstain and write one short, neutral web-search query designed to find
a source that would settle the proposition. Do not write a query that assumes
the proposition is true or false.

PROPOSITION: {proposition}

{rendered}

Return exactly these lines:
{candidate_lines}
SELECTED_ID: <candidate number or NONE>
DECISION: <DECISIVE_SUPPORT or DECISIVE_CONTRADICTION or ABSTAIN>
QUERY: <NONE when decisive, otherwise one neutral search query>
RATIONALE: <one short sentence grounded only in the selected candidate(s)>"""


def parse_audit_completion(
    completion: str,
    *,
    candidates: list[dict[str, Any]],
    proposition: str,
) -> dict[str, Any]:
    assessments = {
        int(position): relation.upper()
        for position, relation in CANDIDATE_RE.findall(completion)
        if 1 <= int(position) <= len(candidates)
    }
    selected_match = SELECTED_RE.search(completion)
    decision_match = DECISION_RE.search(completion)
    query_match = QUERY_RE.search(completion)
    rationale_match = RATIONALE_RE.search(completion)
    decision = decision_match.group(1).upper() if decision_match else "ABSTAIN"
    selected_id = None
    if selected_match and selected_match.group(1).upper() != "NONE":
        selected_id = int(selected_match.group(1))
    valid = (
        len(assessments) == len(candidates)
        and decision_match is not None
        and selected_match is not None
    )
    if decision in DECISIVE:
        valid = (
            valid
            and selected_id is not None
            and 1 <= selected_id <= len(candidates)
            and assessments.get(selected_id) == decision
        )
    else:
        valid = valid and selected_id is None
    if not valid:
        decision = "ABSTAIN"
        selected_id = None
    query = (query_match.group(1).strip() if query_match else "")
    if decision == "ABSTAIN" and (not query or query.upper() == "NONE"):
        query = proposition
    if decision in DECISIVE:
        query = ""
    selected = candidates[selected_id - 1] if selected_id is not None else None
    return {
        "candidate_assessments": [
            {"candidate_id": position, "relation": assessments.get(position)}
            for position in range(1, len(candidates) + 1)
        ],
        "selected_id": selected_id,
        "selected_candidate": selected,
        "decision": decision,
        "query": query,
        "rationale": rationale_match.group(1).strip() if rationale_match else "",
        "parse_valid": bool(valid),
    }
