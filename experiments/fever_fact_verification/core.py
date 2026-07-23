"""Shared parsing and aggregation for the FEVER-style verifier."""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Any, Iterable


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
SPACE = re.compile(r"\s+")
SECTION_HEADING = re.compile(r"\s*=+\s*[^=]+?\s*=+\s*")
TOKEN = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "in", "is", "it",
    "its", "of", "on", "or", "she", "that", "the", "their", "there",
    "they", "this", "to", "was", "were", "which", "who", "with",
}


@dataclass(frozen=True)
class ClaimKey:
    dataset: str
    index: Any
    claim_index: int


def normalize_text(value: str) -> str:
    return SPACE.sub(" ", html.unescape(value)).strip()


def quote_is_grounded(quote: str, output: str) -> bool:
    """Return whether the claimed provenance is an exact output substring."""
    return bool(quote) and quote in output


def last_user_message(prompt: str) -> str:
    """Extract the final user turn from the teacher prompt's context block."""
    context = re.search(r"<context>\s*(.*?)\s*</context>", prompt, re.DOTALL)
    if not context:
        return ""
    matches = re.findall(
        r"(?:^|\n)USER:\s*(.*?)(?=\n(?:SYSTEM|USER|ASSISTANT):|\Z)",
        context.group(1),
        re.DOTALL,
    )
    return normalize_text(matches[-1]) if matches else ""


def split_sentences(text: str, *, minimum_chars: int = 24) -> list[str]:
    """Split a MediaWiki extract while retaining compact factual sentences."""
    text = SECTION_HEADING.sub(" ", text)
    text = normalize_text(text)
    if not text:
        return []
    sentences = SENTENCE_BOUNDARY.split(text)
    return [sentence for sentence in sentences if len(sentence) >= minimum_chars]


def lexical_relevance(query: str, sentence: str) -> float:
    """Score a sentence cheaply before neural reranking of full documents.

    The score intentionally rewards exact numbers because dates and quantities
    are common poisoned details and generic dense similarity can overlook them.
    """
    query_tokens = [token.casefold() for token in TOKEN.findall(query)]
    sentence_tokens = {token.casefold() for token in TOKEN.findall(sentence)}
    content = {token for token in query_tokens if token not in STOPWORDS}
    if not content:
        return 0.0
    overlap = len(content & sentence_tokens) / len(content)
    numbers = {token for token in content if any(character.isdigit() for character in token)}
    number_match = len(numbers & sentence_tokens) / len(numbers) if numbers else 0.0
    return overlap + 0.5 * number_match


def select_document_sentences(
    query: str,
    sentences: list[str],
    *,
    limit: int,
    keep_lead: int = 2,
) -> list[tuple[int, str, float]]:
    """Select bounded full-document evidence while retaining sentence offsets."""
    if limit < 1:
        return []
    scored = [
        (index, sentence, lexical_relevance(query, sentence))
        for index, sentence in enumerate(sentences)
    ]
    lead_indices = set(range(min(keep_lead, limit, len(scored))))
    ranked = sorted(scored, key=lambda item: (item[2], -item[0]), reverse=True)
    selected_indices = set(lead_indices)
    for index, _, _ in ranked:
        if len(selected_indices) >= limit:
            break
        selected_indices.add(index)
    return [item for item in scored if item[0] in selected_indices]


def deduplicate_passages(passages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate title/sentence candidates without changing rank order."""
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for passage in passages:
        title = normalize_text(str(passage.get("title", "")))
        text = normalize_text(str(passage.get("text", "")))
        key = (title.casefold(), text.casefold())
        if not text or key in seen:
            continue
        seen.add(key)
        output.append({**passage, "title": title, "text": text})
    return output


def load_grounded_claims(
    records: Iterable[dict[str, Any]],
    *,
    variant: str = "material_assessed",
    dataset_name_contains: str | None = None,
) -> list[dict[str, Any]]:
    """Flatten grounded atomic claims from the reviewed GPT-OSS schema."""
    output: list[dict[str, Any]] = []
    seen: set[ClaimKey] = set()
    for record in records:
        if str(record.get("variant")) != variant:
            continue
        dataset = str(record["dataset"])
        if dataset_name_contains and dataset_name_contains not in dataset:
            continue
        assistant_output = str(record.get("output") or "")
        prompt = str(record.get("prompt") or "")
        if not assistant_output:
            match = re.search(r"<output>\s*(.*?)\s*</output>", prompt, re.DOTALL)
            assistant_output = match.group(1) if match else ""
        for position, claim in enumerate(record.get("claims") or []):
            quote = str(claim.get("quote") or "").strip()
            proposition = normalize_text(str(claim.get("proposition") or ""))
            if not proposition or not quote_is_grounded(quote, assistant_output):
                continue
            key = ClaimKey(dataset, record["index"], position)
            if key in seen:
                raise ValueError(f"duplicate claim key: {key}")
            seen.add(key)
            output.append({
                "dataset": dataset,
                "index": record["index"],
                "claim_index": position,
                "label": int(record["label"]),
                "quote": quote,
                "proposition": proposition,
                "question": last_user_message(prompt),
                "teacher_assessment": str(claim.get("assessment") or "uncertain"),
            })
    return output


def verdict_from_probabilities(
    entailment: float,
    neutral: float,
    contradiction: float,
    *,
    minimum_confidence: float = 0.5,
) -> str:
    """Map FEVER-NLI probabilities to SUPPORTS, REFUTES, or NEI."""
    if max(entailment, contradiction) < minimum_confidence:
        return "NOT_ENOUGH_INFO"
    if neutral >= max(entailment, contradiction):
        return "NOT_ENOUGH_INFO"
    return "SUPPORTS" if entailment >= contradiction else "REFUTES"


def aggregate_evidence(
    candidates: list[dict[str, Any]],
    *,
    minimum_confidence: float = 0.5,
    top_k: int = 5,
) -> dict[str, Any]:
    """Rank by non-neutral probability and choose the strongest polar evidence."""
    ranked = sorted(
        candidates,
        key=lambda item: (
            1.0 - float(item["neutral"]),
            max(float(item["entailment"]), float(item["contradiction"])),
        ),
        reverse=True,
    )
    selected = ranked[:top_k]
    if not selected:
        return {
            "verdict": "NOT_ENOUGH_INFO",
            "confidence": 0.0,
            "evidence": [],
        }
    best_support = max(selected, key=lambda item: float(item["entailment"]))
    best_refute = max(selected, key=lambda item: float(item["contradiction"]))
    support = float(best_support["entailment"])
    refute = float(best_refute["contradiction"])
    neutral = min(float(item["neutral"]) for item in selected)
    verdict = verdict_from_probabilities(
        support,
        neutral,
        refute,
        minimum_confidence=minimum_confidence,
    )
    return {
        "verdict": verdict,
        "confidence": max(support, refute) if verdict != "NOT_ENOUGH_INFO" else 0.0,
        "support_score": support,
        "refute_score": refute,
        "evidence": selected,
    }
