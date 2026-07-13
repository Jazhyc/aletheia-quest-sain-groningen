#!/usr/bin/env python3
"""Rule-based material-claim extraction and abstaining Wikidata retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import sqlite3
from typing import Any, Iterable

from experiments.wikidata_rag.evaluate_offline_index import fts_expression
from experiments.wikidata_rag.evaluate_retrieval import STOPWORDS, TOKEN_RE


TURN_RE = re.compile(
    r"(?:^|\n)(SYSTEM|USER|ASSISTANT):\s*(.*?)(?=\n(?:SYSTEM|USER|ASSISTANT):|\Z)",
    flags=re.DOTALL,
)
MARKED_RE = re.compile(
    r"\*\*([^*]{2,120})\*\*|(?<!\*)\*([^*]{2,120})\*(?!\*)|[\"“]([^\"”]{2,120})[\"”]|"
    r"(?<!\w)'([^'\n]{3,120})'(?!\w)"
)
PROPER_RE = re.compile(
    r"\b(?:[A-Z][\w’'-]*)(?:\s+(?:of|the|de|del|van|von|and|[A-Z][\w’'-]*)){0,6}"
)
YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
UNSUPPORTED_RELATION_RE = re.compile(
    r"\b(border(?:s|ed|ing)?|flows? (?:into|through)|mouth of|near(?:est)?|distance|"
    r"nickname|moniker|catchphrase|without a safety line|untethered|immediately (?:to the )?"
    r"(?:north|south|east|west)|built to house|destination|air disaster|gained (?:its )?"
    r"independence|what year did .{0,80}\bbecome|belonged to)\b",
    re.I,
)


RELATION_PATTERNS: tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...] = (
    (
        "location",
        re.compile(
            r"(?:(?:^|[?.]\s*)where\b|\b(?:which|what) (?:modern day )?(?:country|county|city|town|island|state|"
            r"province|region|continent)|in (?:which|what) (?:country|county|city|town|island|"
            r"state|province|region|continent)|\b(?:located|location|capital|headquarters|born|"
            r"birthplace|died|death place|set in)\b)",
            re.I,
        ),
        (
            "country", "continent", "capital", "headquarters", "located in", "location",
            "country of origin", "place of birth", "place of death", "location of formation",
            "narrative location", "filming location", "work location", "residence",
        ),
    ),
    (
        "date",
        re.compile(
            r"(?:^|[?.]\s*)when\b|\b(what year|which year|date|born|birth|died|death|founded|formed|"
            r"established|began|started|ended|published|released|opened|closed)\b",
            re.I,
        ),
        (
            "date of birth", "date of death", "inception", "publication date", "start time",
            "end time", "point in time", "dissolved or abolished",
        ),
    ),
    (
        "author",
        re.compile(r"\b(who (?:wrote|authored)|writer|author|written by|novelist|playwright)\b", re.I),
        ("author", "creator"),
    ),
    (
        "director",
        re.compile(r"\b(who directed|director|directed by|filmmaker)\b", re.I),
        ("director",),
    ),
    (
        "creator",
        re.compile(r"\b(who (?:created|invented|discovered|developed)|creator|inventor|discoverer)\b", re.I),
        ("creator", "discoverer or inventor", "developer"),
    ),
    (
        "founder",
        re.compile(r"\b(who founded|founder|founded by|established by)\b", re.I),
        ("founded by",),
    ),
    (
        "performer",
        re.compile(r"\b(who (?:sang|performed|recorded)|singer|performer|recorded by)\b", re.I),
        ("performer",),
    ),
    (
        "cast",
        re.compile(r"\b(who (?:starred|played|portrayed)|cast|actor|actress|starring)\b", re.I),
        ("cast member",),
    ),
    (
        "membership",
        re.compile(r"\b(member|team|club|band|political party|played for)\b", re.I),
        ("member of", "member of sports team", "political party", "has part"),
    ),
    (
        "language",
        re.compile(r"\b(language|spoken|translation|english name)\b", re.I),
        ("official language", "original language"),
    ),
    (
        "category",
        re.compile(r"\b(what (?:is|are|was|were)|which (?:animal|bird|mammal|plant|sport|weapon)|"
                   r"type of|kind of|variet(?:y|ies)|known as|called|name of)\b", re.I),
        ("instance of", "subclass of", "named after", "occupation", "field of work", "genre"),
    ),
)


GENERIC_SPANS = {
    "assistant", "system", "user", "england", "english", "united states",
    "which", "what", "where", "when", "who", "the", "a", "an",
}
LEADING_QUESTION_WORDS = {
    "according", "before", "in", "name", "on", "prior", "published", "that",
    "these", "this", "what", "when", "where", "which", "who",
}


@dataclass(frozen=True)
class ClaimQuery:
    question: str
    answer: str
    subjects: tuple[str, ...]
    answer_values: tuple[str, ...]
    relations: tuple[str, ...]
    predicates: tuple[str, ...]
    qualifiers: tuple[str, ...]


TYPE_HINTS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:television|tv) series\b", re.I), ("television series", "tv series")),
    (re.compile(r"\bfilm\b", re.I), ("film",)),
    (re.compile(r"\bnovel\b", re.I), ("novel", "literary work")),
    (re.compile(r"\bplay\b", re.I), ("play", "theatrical work")),
    (re.compile(r"\b(?:song|theme tune)\b", re.I), ("song", "musical work", "composition")),
)


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower().replace("’", "'")))


def normalize_name(text: str) -> str:
    normalized = normalize(text)
    for prefix in ("the ",):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized


def content_tokens(text: str) -> set[str]:
    return {
        token.lower().replace("’", "'")
        for token in TOKEN_RE.findall(text)
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }


def parse_final_exchange(conversation: str) -> tuple[str, str]:
    turns: dict[str, list[str]] = {"USER": [], "ASSISTANT": []}
    for role, text in TURN_RE.findall(conversation):
        if role in turns:
            turns[role].append(" ".join(text.split()))
    question = turns["USER"][-1] if turns["USER"] else ""
    answer = turns["ASSISTANT"][-1] if turns["ASSISTANT"] else ""
    return question, answer


def marked_spans(text: str) -> list[str]:
    return [
        next(group for group in match.groups() if group).strip(" .,:;!?()[]")
        for match in MARKED_RE.finditer(text)
    ]


def proper_spans(text: str) -> list[str]:
    spans = []
    for match in PROPER_RE.finditer(text):
        span = match.group(0).strip(" .,:;!?()[]")
        normalized = normalize(span)
        first = normalized.partition(" ")[0]
        if (
            normalized not in GENERIC_SPANS
            and first not in LEADING_QUESTION_WORDS
            and len(normalized) >= 3
        ):
            spans.append(span)
    return spans


def unique_spans(spans: Iterable[str], limit: int = 12) -> tuple[str, ...]:
    selected: list[str] = []
    seen: set[str] = set()
    for span in spans:
        key = normalize(span)
        if not key or key in seen or len(key) > 120:
            continue
        seen.add(key)
        selected.append(span)
        if len(selected) == limit:
            break
    return tuple(selected)


def refine_predicates(question: str, predicates: list[str]) -> list[str]:
    """Narrow broad location/date groups to the relation actually requested."""
    if UNSUPPORTED_RELATION_RE.search(question):
        return []
    location: tuple[tuple[str, tuple[str, ...]], ...] = (
        (r"\b(born|birthplace|place of birth)\b", ("place of birth",)),
        (r"\b(died|place of death)\b", ("place of death",)),
        (r"\bheadquarters?\b", ("headquarters",)),
        (r"\bcapital\b", ("capital",)),
        (r"\b(country|nation)\b", ("country", "country of origin", "located in")),
        (r"\bcontinent\b", ("continent",)),
        (r"\b(city|town|island)\b", ("located in", "location", "headquarters", "capital")),
        (r"\b(county|state|province|region)\b", ("located in", "location")),
        (r"\b(set|filmed|takes place)\b", ("narrative location", "filming location", "location")),
    )
    date: tuple[tuple[str, tuple[str, ...]], ...] = (
        (r"\b(born|birth)\b", ("date of birth",)),
        (r"\b(died|death)\b", ("date of death",)),
        (r"\b(published|publication|released)\b", ("publication date",)),
        (r"\b(founded|formed|established|opened)\b", ("inception", "start time")),
        (r"\b(ended|closed|dissolved)\b", ("end time", "dissolved or abolished")),
    )
    narrowed = list(predicates)
    if "country" in predicates or "located in" in predicates:
        matches = [values for pattern, values in location if re.search(pattern, question, re.I)]
        if matches:
            location_all = set(RELATION_PATTERNS[0][2])
            narrowed = [value for value in narrowed if value not in location_all]
            narrowed.extend(value for values in matches for value in values)
    if "date of birth" in predicates or "publication date" in predicates:
        matches = [values for pattern, values in date if re.search(pattern, question, re.I)]
        if matches:
            date_all = set(RELATION_PATTERNS[1][2])
            narrowed = [value for value in narrowed if value not in date_all]
            narrowed.extend(value for values in matches for value in values)
    return list(dict.fromkeys(narrowed))


def extract_claim_query(conversation: str) -> ClaimQuery:
    """Extract query entities, expected predicates, and important qualifiers."""
    question, answer = parse_final_exchange(conversation)
    relation_names: list[str] = []
    predicates: list[str] = []
    for name, pattern, matched_predicates in RELATION_PATTERNS:
        if pattern.search(question):
            relation_names.append(name)
            predicates.extend(matched_predicates)
    predicates = refine_predicates(question, predicates)

    first_answer = re.split(r"(?<=[.!?])\s+", answer, maxsplit=1)[0]
    question_marked = marked_spans(question)
    answer_marked = marked_spans(first_answer)
    question_proper = proper_spans(question)
    answer_proper = proper_spans(first_answer)

    # The subject usually lives in the question (the work, person, or place being
    # queried). Answer entities are useful fallbacks for identity questions but
    # are deliberately ranked later to avoid retrieving only the poisoned value.
    subjects = unique_spans(question_marked + question_proper)
    answer_values = unique_spans(answer_marked + answer_proper, limit=6)
    qualifiers = unique_spans(
        YEAR_RE.findall(question + " " + first_answer)
        + re.findall(
            r"\b(first|last|only|without|before|after|between|longest|shortest|highest|lowest|"
            r"largest|smallest|most|least|never|not)\b",
            question + " " + first_answer,
            flags=re.I,
        ),
        limit=8,
    )
    return ClaimQuery(
        question=question,
        answer=first_answer,
        subjects=subjects,
        answer_values=answer_values,
        relations=tuple(relation_names),
        predicates=tuple(
            predicate
            for predicate in dict.fromkeys(predicates)
            if not ("category" in relation_names and len(relation_names) > 1
                    and predicate in RELATION_PATTERNS[-1][2])
        ),
        qualifiers=qualifiers,
    )


def split_facts(raw_facts: str | Iterable[str]) -> list[str]:
    if isinstance(raw_facts, str):
        return [fact.strip() for fact in raw_facts.split("; ") if fact.strip()]
    return [str(fact).strip() for fact in raw_facts if str(fact).strip()]


def fact_predicate(fact: str) -> str:
    return fact.partition(":")[0].strip().lower()


def entity_name_score(entity: dict[str, Any], claim: ClaimQuery) -> tuple[int, str | None]:
    names = [entity.get("label", ""), *(entity.get("aliases") or [])]
    normalized_names = [(name, normalize(name)) for name in names if normalize(name)]
    best = (0, None)
    for position, subject in enumerate(claim.subjects[:1]):
        subject_norm = normalize_name(subject)
        subject_tokens = content_tokens(subject)
        for name, name_norm in normalized_names:
            name_norm = normalize_name(name)
            if subject_norm == name_norm:
                score = 100 - position
            else:
                overlap = len(subject_tokens & content_tokens(name))
                score = (20 + 10 * overlap - position) if overlap else 0
            if score > best[0]:
                best = (score, name)
    return best


def gate_entity(entity: dict[str, Any], claim: ClaimQuery) -> dict[str, Any] | None:
    """Keep only facts that align with both a named subject and question relation."""
    name_score, matched_name = entity_name_score(entity, claim)
    if name_score < 70:
        return None
    entity_type_text = " ".join([
        entity.get("description", ""), *split_facts(entity.get("facts", [])),
    ]).lower()
    for pattern, expected in TYPE_HINTS:
        if pattern.search(claim.question) and not any(term in entity_type_text for term in expected):
            return None
    facts = split_facts(entity.get("facts", []))
    relevant_facts = [
        fact for fact in facts if fact_predicate(fact) in set(claim.predicates)
    ]
    if not relevant_facts:
        return None
    return {
        **entity,
        "facts": relevant_facts,
        "gate": {
            "matched_name": matched_name,
            "name_score": name_score,
            "relations": list(claim.relations),
            "predicates": sorted({fact_predicate(fact) for fact in relevant_facts}),
        },
    }


def entity_context_score(entity: dict[str, Any], claim: ClaimQuery) -> int:
    """Disambiguate exact-name entities using question context and stated value."""
    subject_tokens = set().union(*(content_tokens(span) for span in claim.subjects))
    question_context = content_tokens(claim.question) - subject_tokens
    entity_text = " ".join([
        entity.get("description", ""),
        *split_facts(entity.get("facts", [])),
    ])
    context_overlap = len(question_context & content_tokens(entity_text))
    answer_tokens = set().union(
        *(content_tokens(value) for value in claim.answer_values)
    ) if claim.answer_values else set()
    relevant_values = " ".join(
        fact.partition(":")[2]
        for fact in split_facts(entity.get("facts", []))
        if fact_predicate(fact) in set(claim.predicates)
    )
    value_overlap = len(answer_tokens & content_tokens(relevant_values))
    return 12 * value_overlap + 3 * context_overlap


def query_candidate_entities(
    connection: sqlite3.Connection,
    claim: ClaimQuery,
    *,
    per_query: int = 32,
) -> list[dict[str, Any]]:
    """Retrieve exact-name candidates for question subjects and answer fallbacks."""
    sql = """
        SELECT e.qid, e.label, e.aliases, e.description, e.facts,
               e.popularity, bm25(entity_fts, 8.0, 5.0, 1.0, 1.0) AS rank
        FROM entity_fts
        JOIN entity AS e ON e.rowid = entity_fts.rowid
        WHERE entity_fts MATCH ?
        ORDER BY rank, e.popularity DESC
        LIMIT ?
    """
    selected: dict[str, dict[str, Any]] = {}
    for span in claim.subjects:
        expression = fts_expression(span, phrase=True, max_tokens=10)
        if not expression:
            continue
        try:
            rows = connection.execute(sql, (expression, per_query)).fetchall()
        except sqlite3.OperationalError:
            continue
        for row in rows:
            entity = {
                "qid": row[0], "title": row[1], "label": row[1],
                "aliases": row[2].split("; ") if row[2] else [],
                "description": row[3], "facts": split_facts(row[4]),
                "popularity": row[5], "rank": row[6],
            }
            score, _ = entity_name_score(entity, claim)
            if score >= 70:
                selected.setdefault(entity["qid"], entity)
    return list(selected.values())


def retrieve_claim_evidence(
    connection: sqlite3.Connection,
    conversation: str,
    *,
    limit: int = 3,
) -> dict[str, Any]:
    """Return a structured claim and zero or more relation-gated entity cards."""
    claim = extract_claim_query(conversation)
    if claim.relations == ("category",):
        return {"claim": asdict(claim), "entities": [], "abstain_reason": "unsupported_category"}
    if not claim.predicates or not claim.subjects:
        return {"claim": asdict(claim), "entities": [], "abstain_reason": "no_relation_or_subject"}
    candidates = query_candidate_entities(connection, claim)
    gated = [entity for entity in (gate_entity(row, claim) for row in candidates) if entity]
    primary_tokens = content_tokens(claim.subjects[0])
    if len(primary_tokens) == 1 and len(claim.subjects) > 1:
        gated = [row for row in gated if entity_context_score(row, claim) > 0]
    gated.sort(
        key=lambda row: (
            -int(row["gate"]["name_score"]),
            -entity_context_score(row, claim),
            -len(row["facts"]),
            -int(row.get("popularity", 0)),
            float(row.get("rank", 0.0)),
        )
    )
    if len(gated) > 1:
        best = gated[0]
        runner_up = gated[1]
        same_name_tier = best["gate"]["name_score"] == runner_up["gate"]["name_score"]
        context_margin = entity_context_score(best, claim) - entity_context_score(runner_up, claim)
        if same_name_tier and context_margin < 3:
            return {"claim": asdict(claim), "entities": [], "abstain_reason": "ambiguous_entity"}
    return {
        "claim": asdict(claim),
        "entities": gated[:limit],
        "abstain_reason": None if gated else "no_relation_aligned_fact",
    }
