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
    r"\b(?:[A-Z][\w’'-]*)(?:\s+(?:[A-Z][\w’'-]*|"
    r"(?:(?:of|in|the|de|del|della|van|von|and)\s+)+[A-Z][\w’'-]*)){0,6}"
)
YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
UNSUPPORTED_RELATION_RE = re.compile(
    r"\b(flows? (?:into|through)|mouth of|near(?:est)?|distance|"
    r"catchphrase|without a safety line|untethered|immediately (?:to the )?"
    r"(?:north|south|east|west)|built to house|destination|air disaster|gained (?:its )?"
    r"independence|what year did .{0,80}\bbecome|belonged to)\b",
    re.I,
)
UNSUPPORTED_SLOT_RE = re.compile(
    r"\b(?:what|which) (?:battle)?ship (?:was|were) sunk\b|"
    r"\bwhich river forms? (?:the )?border\b|"
    r"\b(?:olympic )?(?:bronze|silver|gold) medal in which (?:athletics )?event\b|"
    r"\bduring the reign of which monarch\b|"
    r"\bwhich sport was played .{0,80}\bprior to\b",
    re.I,
)


RELATION_PATTERNS: tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...] = (
    (
        "location",
        re.compile(
            r"(?:(?:^|[?.]\s*)where\b|\b(?:which|what) (?:modern[ -]day )?(?:country|county|city|town|island|state|"
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
        "screenwriter",
        re.compile(r"\b(who (?:wrote the (?:screenplay|script)|scripted)|screenwriter|screenplay|script by)\b", re.I),
        ("screenwriter",),
    ),
    (
        "composer",
        re.compile(r"\b(who composed|composer|composed by|music by|lyrics? by|lyricist|wrote the lyrics)\b", re.I),
        ("composer", "lyrics by"),
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
        re.compile(r"\b(who (?:starred|played|portrayed|voiced)|cast|actor|actress|starring|character appears)\b", re.I),
        ("cast member", "voice actor", "characters"),
    ),
    (
        "border",
        re.compile(r"\b(border(?:s|ed|ing)?|shares? (?:a )?border|neighbou?rs?)\b", re.I),
        ("shares border with",),
    ),
    (
        "ownership",
        re.compile(r"\b(who (?:owns|owned|operates)|owned by|owner|operator|parent (?:company|organization)|subsidiar(?:y|ies)|chief executive|\bceo\b|chairperson)\b", re.I),
        ("owned by", "operator", "parent organization", "subsidiary", "chief executive officer", "chairperson"),
    ),
    (
        "leadership",
        re.compile(r"\b(head of (?:state|government)|president|prime minister|military rank|position held|which position did .{0,80}\bhold)\b", re.I),
        ("head of state", "head of government", "position held", "military rank"),
    ),
    (
        "sport",
        re.compile(r"\b(which|what) (?:sport|league|position)|\b(sport|league|position played|played as)\b", re.I),
        ("sport", "league", "position played", "country for sport"),
    ),
    (
        "event",
        re.compile(r"\b(participat(?:e|ed|es|ing|ion)|participant|competed|competition|conflict|war|battle|organizer|organised|organized|winners?|won|major event|event involved)\b", re.I),
        ("participant", "participant in", "conflict", "organizer", "significant event", "winner"),
    ),
    (
        "award",
        re.compile(
            r"\b(?:which|what) awards? did .{0,100}\b(?:win|receive)|\baward received\b|"
            r"\bwon .{0,100}\b(?:oscar|award).{0,100}\bfor which (?:film|movie|role|work)",
            re.I,
        ),
        ("award received",),
    ),
    (
        "cause",
        re.compile(r"\b(cause[sd]? (?:of|by)|caused by|cause of death|manner of death|how did .{0,60}\bdie)\b", re.I),
        ("has cause", "cause of death", "manner of death"),
    ),
    (
        "name",
        re.compile(r"\b(official name|native name|short name|nickname|moniker|also known as|same as|different from|local name|locals call)\b", re.I),
        ("alias", "official name", "name in native language", "short name", "nickname", "name", "said to be the same as", "different from"),
    ),
    (
        "membership",
        re.compile(r"\b(member|team|club|band|political party|played for)\b", re.I),
        ("member of", "member of sports team", "political party", "has part"),
    ),
    (
        "language",
        re.compile(r"\b(language|spoken|translation|english name)\b", re.I),
        ("official language", "original language", "native language", "language of work", "languages spoken"),
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
    (re.compile(r"\b(?:play|playwright)\b", re.I), ("play", "theatrical work")),
    (re.compile(r"\b(?:song|theme tune)\b", re.I), ("song", "musical work", "composition")),
)

TEMPORAL_PREDICATES = {
    "head of government", "head of state", "organizer", "participant",
    "participant in", "position held", "significant event", "winner",
}


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower().replace("’", "'")))


def normalize_name(text: str) -> str:
    normalized = normalize(text)
    if normalized.startswith("st "):
        normalized = "saint " + normalized[3:]
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


def augment_year_event_subjects(subjects: tuple[str, ...], question: str) -> tuple[str, ...]:
    """Prefer edition entities when a named recurring event has an explicit year."""
    years = YEAR_RE.findall(question)
    if not years:
        return subjects
    augmented = []
    for subject in subjects:
        if (
            re.search(r"\b(?:world cup|olympics?|championships?|games)\b", subject, re.I)
            and not YEAR_RE.search(subject)
        ):
            augmented.append(f"{years[0]} {subject}")
        augmented.append(subject)
    return unique_spans(augmented)


def refine_predicates(question: str, predicates: list[str]) -> list[str]:
    """Narrow broad location/date groups to the relation actually requested."""
    if UNSUPPORTED_RELATION_RE.search(question) or UNSUPPORTED_SLOT_RE.search(question):
        return []
    location: tuple[tuple[str, tuple[str, ...]], ...] = (
        (r"\b(born|birthplace|place of birth)\b", ("place of birth",)),
        (r"\b(died|place of death)\b", ("place of death",)),
        (r"\bheadquarters?\b", ("headquarters",)),
        (r"\bcapital\b", ("capital",)),
        (r"\b(country|nation)\b", ("country", "country of origin", "located in")),
        (r"\bcontinent\b", ("continent",)),
        (r"\b(?:on )?(?:which|what) island\b", ("located on physical feature", "located in", "location")),
        (r"\b(city|town)\b", ("located in", "location", "headquarters")),
        (r"\b(tomb|burial|buried)\b", ("place of burial", "located in", "country")),
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
    event_all = set(next(values for name, _, values in RELATION_PATTERNS if name == "event"))
    if event_all & set(narrowed):
        event_values: tuple[str, ...]
        if re.search(r"\b(winners?|won)\b", question, re.I):
            event_values = ("winner",)
        elif re.search(r"\b(participated|participant|competed)\b", question, re.I):
            event_values = ("participant", "participant in")
        elif re.search(r"\b(organizer|organised|organized)\b", question, re.I):
            event_values = ("organizer",)
        elif re.search(r"\b(conflict|war|battle)\b", question, re.I):
            event_values = () if re.search(
                r"(?:^|[?.]\s*)(?:where|in (?:what|which) (?:country|place|region))\b",
                question,
                re.I,
            ) else ("conflict", "significant event")
        else:
            event_values = tuple(event_all)
        narrowed = [value for value in narrowed if value not in event_all]
        narrowed.extend(event_values)
    sport_all = {"sport", "league", "position played", "country for sport"}
    if sport_all & set(narrowed):
        if re.search(r"\bwho\b.{0,100}\bwinners?\b", question, re.I):
            sport_values: tuple[str, ...] = ()
        elif re.search(r"\b(which|what) (?:sport)|injury in which sport\b", question, re.I):
            sport_values = ("sport", "country for sport")
        elif re.search(r"\bleague\b", question, re.I):
            sport_values = ("league",)
        elif re.search(r"\b(position|played as)\b", question, re.I):
            sport_values = ("position played",)
        else:
            sport_values = tuple(sport_all)
        narrowed = [value for value in narrowed if value not in sport_all]
        narrowed.extend(sport_values)
    if "winner" in narrowed:
        membership_all = {"member of", "member of sports team", "political party", "has part"}
        narrowed = [value for value in narrowed if value not in membership_all]
        if re.search(r"\bwhich country's\b", question, re.I):
            location_all = set(RELATION_PATTERNS[0][2])
            narrowed = [value for value in narrowed if value not in location_all]
    cast_all = {"cast member", "voice actor", "characters"}
    if cast_all & set(narrowed):
        if re.search(r"\bvoice(?:d| actor)?\b", question, re.I):
            cast_values = ("voice actor",)
        elif re.search(r"\bcharacter\b", question, re.I) and not re.search(
            r"\bwho (?:played|portrayed|starred)\b", question, re.I
        ):
            cast_values = ("characters",)
        else:
            cast_values = ("cast member",)
        narrowed = [value for value in narrowed if value not in cast_all]
        narrowed.extend(cast_values)
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

    answer_sentences = re.split(r"(?<=[.!?])\s+", answer)
    first_answer = answer_sentences[0]
    # Polished/spokesperson responses often spend the first sentence on a
    # preamble. Scan a short bounded prefix for the actual proposed value while
    # retaining the first sentence as the compact claim text.
    answer_scan = " ".join(answer_sentences[:3])[:600]
    question_marked = marked_spans(question)
    answer_marked = marked_spans(answer_scan)
    question_proper = proper_spans(question)
    answer_proper = proper_spans(answer_scan)

    # The subject usually lives in the question (the work, person, or place being
    # queried). Answer entities are useful fallbacks for identity questions but
    # are deliberately ranked later to avoid retrieving only the poisoned value.
    subjects = augment_year_event_subjects(
        unique_spans(question_marked + question_proper), question
    )
    answer_values = unique_spans(answer_marked + answer_proper, limit=6)
    qualifiers = unique_spans(
        YEAR_RE.findall(question + " " + answer_scan)
        + re.findall(
            r"\b(first|last|only|without|before|after|between|longest|shortest|highest|lowest|"
            r"largest|smallest|most|least|never|not)\b",
            question + " " + answer_scan,
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


def temporally_compatible(fact: str, claim: ClaimQuery) -> bool:
    """Reject explicitly dated facts that conflict with an explicitly dated query."""
    if fact_predicate(fact) not in TEMPORAL_PREDICATES:
        return True
    query_years = set(YEAR_RE.findall(claim.question))
    fact_years = set(YEAR_RE.findall(fact))
    return not query_years or not fact_years or bool(query_years & fact_years)


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
    if re.search(r"\b(?:patron )?saint\b|\bsaint'?s tomb\b", claim.question, re.I):
        instance_text = " ".join(
            fact.partition(":")[2]
            for fact in facts
            if fact_predicate(fact) in {"instance of", "subclass of"}
        ).lower()
        if not any(term in instance_text for term in ("human", "saint", "martyr")):
            return None
    if re.search(r"\bwhat island\b", claim.question, re.I):
        instance_text = " ".join(
            fact.partition(":")[2]
            for fact in facts
            if fact_predicate(fact) in {"instance of", "subclass of"}
        ).lower()
        if "island" not in instance_text:
            return None
    relevant_facts = [
        fact for fact in facts
        if fact_predicate(fact) in set(claim.predicates)
        and temporally_compatible(fact, claim)
    ]
    if (
        "alias" in claim.predicates
        and matched_name
        and normalize_name(matched_name) != normalize_name(entity.get("label", ""))
    ):
        relevant_facts.append(f"alias: {matched_name}")
    description = str(entity.get("description", ""))
    if (
        re.search(r"\b(?:on )?(?:which|what) island\b", claim.question, re.I)
        and re.search(r"\bon (?:the )?island of\b", description, re.I)
    ):
        relevant_facts.append(f"description: {description}")
    if re.search(r"\bmodern[ -]day country\b", claim.question, re.I):
        current_country = [
            fact for fact in relevant_facts
            if fact_predicate(fact) == "country" and "[" not in fact
        ]
        if current_country:
            relevant_facts = current_country
    if re.search(r"\b(?:which|what) country\b|\bin which country\b", claim.question, re.I):
        direct_country = [
            fact for fact in relevant_facts
            if fact_predicate(fact) in {"country", "country of origin"}
        ]
        if direct_country:
            query_years = set(YEAR_RE.findall(claim.question))
            year_matched = [
                fact for fact in direct_country
                if set(YEAR_RE.findall(fact)) & query_years
            ]
            unqualified = [fact for fact in direct_country if "[" not in fact]
            relevant_facts = year_matched or unqualified or direct_country
    if re.search(r"\bwhat island\b", claim.question, re.I):
        alias_facts = [fact for fact in relevant_facts if fact_predicate(fact) == "alias"]
        if alias_facts:
            relevant_facts = alias_facts
    if re.search(r"\b(?:on )?(?:which|what) island\b", claim.question, re.I):
        descriptions = [
            fact for fact in relevant_facts if fact_predicate(fact) == "description"
        ]
        if descriptions:
            relevant_facts = descriptions
    if re.search(r"\b(tomb|burial|buried)\b", claim.question, re.I):
        burial = [fact for fact in relevant_facts if fact_predicate(fact) == "place of burial"]
        if burial:
            relevant_facts = burial
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
        query_names = [span]
        normalized = normalize_name(span)
        if normalized != normalize(span):
            query_names.append(normalized)
        for query_name in query_names:
            expressions = [
                fts_expression(query_name, phrase=True, max_tokens=10),
                fts_expression(query_name, phrase=False, max_tokens=10),
            ]
            selected_before = len(selected)
            for expression in dict.fromkeys(value for value in expressions if value):
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
                if len(selected) > selected_before:
                    break
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
