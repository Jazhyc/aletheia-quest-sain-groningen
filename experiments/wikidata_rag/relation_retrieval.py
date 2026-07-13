#!/usr/bin/env python3
"""Bidirectional, answer-aware retrieval over the compact relation index."""

from __future__ import annotations

from dataclasses import asdict, replace
import re
import sqlite3
from typing import Any, Iterable

from experiments.wikidata_rag.build_relation_index import PREDICATE_IDS, TARGET_PREDICATES
from experiments.wikidata_rag.claim_retrieval import (
    ClaimQuery,
    TYPE_HINTS,
    YEAR_RE,
    fact_predicate,
    entity_context_score,
    extract_claim_query,
    normalize_name,
    query_candidate_entities,
    split_facts,
    temporally_compatible,
    unique_spans,
)


INVERSE_PREDICATES: dict[str, tuple[str, ...]] = {
    "winner": ("award received",),
    "participant": ("participant in",),
    "participant in": ("participant",),
}

# Only use a differing value as counterevidence for relations that are usually
# single-valued in the scope of a question. Multi-valued relations still return
# facts, but remain neutral when the answer is absent.
FUNCTIONAL_PREDICATES = {
    "capital", "country", "country for sport", "country of origin", "creator", "director",
    "head of government", "head of state", "league", "location", "organizer",
    "performer", "place of burial", "sport", "winner",
}

# A conflicting card value is safe enough to expose only for slots that are
# normally single-valued for the entity and time implied by the question.
# Multi-valued/category relations can still provide exact support, but a
# different value is not evidence that the proposed value is false.
CARD_FUNCTIONAL_PREDICATES = {
    "capital", "country", "country for sport", "country of origin", "creator",
    "date of birth", "date of death", "director", "dissolved or abolished",
    "headquarters", "inception", "location of formation", "manner of death",
    "native language", "official language", "original language", "performer",
    "place of birth", "place of death", "publication date", "screenwriter",
}

CARD_COUNTER_PATTERNS: dict[str, re.Pattern[str]] = {
    "capital": re.compile(r"\bcapital\b", re.I),
    "country": re.compile(r"\b(?:which|what|modern[ -]day) country\b|\bin which country\b", re.I),
    "country of origin": re.compile(r"\bcountry of origin\b|\boriginat(?:e|ed|es)\b", re.I),
    "date of birth": re.compile(r"\b(?:when|what year|which year)\b.{0,80}\bborn\b|\bborn\b.{0,80}\b(?:when|what year|which year)\b", re.I),
    "date of death": re.compile(r"\b(?:when|what year|which year)\b.{0,80}\bdie(?:d)?\b|\bdie(?:d)?\b.{0,80}\b(?:when|what year|which year)\b", re.I),
    "director": re.compile(r"\b(?:who directed|director|directed by)\b", re.I),
    "headquarters": re.compile(r"\bheadquarters?\b", re.I),
    "inception": re.compile(r"(?:^|[?.]\s*)when\b.{0,100}\b(?:founded|formed|established|opened)\b|\b(?:what|which) (?:year|decade)\b.{0,100}\b(?:founded|formed|established|opened)\b|\b(?:founded|formed|established|opened)\b.{0,100}\b(?:what|which) (?:year|decade)\b", re.I),
    "location of formation": re.compile(r"\b(?:formed|founded|established)\b.{0,80}\b(?:where|city|country)\b|\b(?:where|city|country)\b.{0,80}\b(?:formed|founded|established)\b", re.I),
    "native language": re.compile(r"\bnative language\b", re.I),
    "official language": re.compile(r"\bofficial language\b", re.I),
    "original language": re.compile(r"\boriginal language\b", re.I),
    "performer": re.compile(r"\b(?:who sang|who performed|performed by|recorded by)\b", re.I),
    "place of birth": re.compile(r"\b(?:where|which|what)\b.{0,80}\b(?:born|birth ?place)\b|\b(?:born|birth ?place)\b.{0,80}\b(?:where|which|what)\b", re.I),
    "place of death": re.compile(r"\b(?:where|which|what)\b.{0,80}\b(?:died|death place)\b|\b(?:died|death place)\b.{0,80}\b(?:where|which|what)\b", re.I),
    "publication date": re.compile(r"\b(?:published|released)\b.{0,80}\b(?:when|what year|which year)\b|\b(?:when|what year|which year)\b.{0,80}\b(?:published|released)\b", re.I),
    "screenwriter": re.compile(r"\b(?:screenwriter|screenplay|script by|wrote the (?:screenplay|script))\b", re.I),
}

CARD_SUPPORT_PATTERNS: dict[str, re.Pattern[str]] = {
    **CARD_COUNTER_PATTERNS,
    "different from": re.compile(r"\b(?:also known as|called|local(?:s)? call|local name)\b", re.I),
    "genre": re.compile(r"\b(?:type|kind|genre) of (?:book|film|music|work)|\bfamous for writing\b", re.I),
    "instance of": re.compile(r"\b(?:what|which) (?:is|are|was|were|type|kind)\b|\btype of\b|\bkind of\b", re.I),
    "located in": re.compile(r"\b(?:where|which|what) (?:city|town|county|state|country|region|province)\b|\bin (?:which|what) (?:city|town|county|state|country|region|province)\b", re.I),
    "narrative location": re.compile(r"\b(?:set|takes place)\b.{0,80}\b(?:where|city|town|country|location)\b|\b(?:where|city|town|country|location)\b.{0,80}\b(?:set|takes place)\b", re.I),
    "nickname": re.compile(r"\bnickname|nicknamed\b", re.I),
    "position held": re.compile(r"\b(?:became|served as|position held|president|prime minister|first minister)\b", re.I),
    "said to be the same as": re.compile(r"\b(?:also known as|called|local(?:s)? call|local name)\b", re.I),
    "short name": re.compile(r"\b(?:short name|also known as|called)\b", re.I),
    "subclass of": re.compile(r"\b(?:what|which) (?:is|are|was|were|type|kind)\b|\btype of\b|\bkind of\b", re.I),
}


def intervals_overlap(left: sqlite3.Row, right: sqlite3.Row) -> bool:
    return (
        left["end_year"] is None or right["start_year"] is None
        or right["start_year"] <= left["end_year"]
    ) and (
        right["end_year"] is None or left["start_year"] is None
        or left["start_year"] <= right["end_year"]
    )


def predicate_name(predicate_id: int) -> str:
    return TARGET_PREDICATES[predicate_id - 1]


def linked_qids(entity_connection: sqlite3.Connection, spans: Iterable[str]) -> list[str]:
    """Resolve exact subject/answer names through the existing alias-rich FTS."""
    values = unique_spans(spans)
    if not values:
        return []
    linking_claim = ClaimQuery(
        question="", answer="", subjects=values, answer_values=(), relations=(),
        predicates=(), qualifiers=(),
    )
    return [row["qid"] for row in query_candidate_entities(entity_connection, linking_claim)]


def linked_subject_qids(entity_connection: sqlite3.Connection, claim: ClaimQuery) -> list[str]:
    """Link subjects with question context and explicit work-type constraints."""
    selected: dict[str, dict[str, Any]] = {}
    subject_spans = claim.subjects
    if (
        len(subject_spans) > 1
        and re.search(r"\bat (?:the )?.+?\bin which city\b", claim.question, re.I)
    ):
        subject_spans = subject_spans[-1:]
    for span in subject_spans:
        span_claim = replace(claim, subjects=(span,))
        candidates = query_candidate_entities(entity_connection, span_claim)
        for pattern, expected in TYPE_HINTS:
            if pattern.search(claim.question):
                typed = [row for row in candidates if any(
                    term in " ".join([row.get("description", ""), *row.get("facts", [])]).lower()
                    for term in expected
                )]
                candidates = typed
        if len(candidates) > 1:
            scores = [entity_context_score(row, claim) for row in candidates]
            best = max(scores)
            if best > 0:
                candidates = [row for row, score in zip(candidates, scores) if score == best]
            if len(candidates) > 1:
                candidates = [max(candidates, key=lambda row: int(row.get("popularity", 0)))]
        for candidate in candidates:
            selected.setdefault(candidate["qid"], candidate)
    return list(selected)


def trusted_subject_qids(
    entity_connection: sqlite3.Connection,
    claim: ClaimQuery,
    subject_qids: list[str],
) -> list[str]:
    """Restrict counterfacts to the grammatical subject, not incidental names."""
    first = set(linked_qids(entity_connection, claim.subjects[:1]))
    trusted = [qid for qid in subject_qids if qid in first]
    coordinated = len(claim.subjects) > 1 and bool(re.search(r"\band\b", claim.question, re.I))
    typed_focus = any(pattern.search(claim.question) for pattern, _ in TYPE_HINTS)
    if not trusted and subject_qids and (coordinated or typed_focus):
        trusted = [subject_qids[0]]
    return trusted


def card_value(fact: str) -> str:
    """Return a rendered card value without its predicate or qualifiers."""
    return fact.partition(":")[2].partition(" [")[0].strip()


def answer_contains_card_value(claim: ClaimQuery, fact: str) -> bool:
    """Require a complete value or explicit year match in the assistant answer."""
    value = card_value(fact)
    if not value:
        return False
    predicate = fact_predicate(fact)
    answer_years = set(YEAR_RE.findall(claim.answer))
    value_years = set(YEAR_RE.findall(value))
    counter_pattern = CARD_COUNTER_PATTERNS.get(predicate)
    if (
        answer_years and value_years and answer_years & value_years
        and counter_pattern and counter_pattern.search(claim.question)
    ):
        return True
    answer_decades = {int(match) * 10 for match in re.findall(r"\b(1\d{2}|20\d)0s\b", claim.answer)}
    if (
        answer_decades and value_years
        and any((int(year) // 10) * 10 in answer_decades for year in value_years)
        and counter_pattern and counter_pattern.search(claim.question)
    ):
        return True
    normalized_value = normalize_name(value)
    if len(normalized_value) < 3:
        return False
    subject_values = {normalize_name(item) for item in claim.subjects}
    normalized_question = normalize_name(claim.question)
    primary_values = [
        normalize_name(item) for item in claim.answer_values
        if normalize_name(item) not in subject_values
        and normalize_name(item) not in normalized_question
    ]
    if primary_values and normalized_value == primary_values[0]:
        return True
    # Lowercase concepts such as "detective fiction" are not named-entity
    # spans. Accept a complete multiword phrase only when it adds information
    # not already repeated from the question.
    if (
        len(normalized_value.split()) < 2 or value[:1].isupper()
        or not value[:1].isalpha()
    ):
        return False
    normalized_answer = normalize_name(claim.answer)
    return (
        normalized_value not in normalized_question
        and bool(re.search(rf"(?:^|\s){re.escape(normalized_value)}(?:$|\s)", normalized_answer))
    )


def retrieve_card_verdict(
    entity_connection: sqlite3.Connection,
    claim: ClaimQuery,
    subject_qids: list[str],
    *,
    limit: int = 6,
) -> dict[str, Any] | None:
    """Classify strictly routed facts already stored in the entity-linker DB."""
    trusted = trusted_subject_qids(entity_connection, claim, subject_qids)
    if not trusted or not claim.predicates:
        return None
    placeholders = ",".join("?" for _ in trusted)
    rows = entity_connection.execute(
        f"SELECT qid,label,facts FROM entity WHERE qid IN ({placeholders})", trusted
    ).fetchall()
    allowed = set(claim.predicates)
    candidates: list[dict[str, str]] = []
    for qid, label, raw_facts in rows:
        for fact in split_facts(raw_facts):
            predicate = fact_predicate(fact)
            if predicate in allowed and temporally_compatible(fact, claim):
                candidates.append({
                    "qid": str(qid), "subject_label": str(label),
                    "predicate": predicate, "fact": fact,
                })
    if not candidates:
        return None
    matches = [
        row for row in candidates
        if row["predicate"] in CARD_SUPPORT_PATTERNS
        and CARD_SUPPORT_PATTERNS[row["predicate"]].search(claim.question)
        and answer_contains_card_value(claim, row["fact"])
    ]
    if matches:
        selected = matches
        status = "support"
    else:
        functional = [
            row for row in candidates
            if row["predicate"] in CARD_FUNCTIONAL_PREDICATES
            and row["predicate"] in CARD_COUNTER_PATTERNS
            and CARD_COUNTER_PATTERNS[row["predicate"]].search(claim.question)
            and not (
                row["predicate"] in {
                    "date of birth", "date of death", "inception", "publication date",
                }
                and YEAR_RE.search(card_value(row["fact"]))
                and int(YEAR_RE.search(card_value(row["fact"])).group()) < 1000
            )
        ]
        # Multiple conflicting values (especially dates and locations) usually
        # signal a historical/ambiguous slot, so abstain rather than cherry-pick.
        distinct = {(row["predicate"], normalize_name(card_value(row["fact"]))) for row in functional}
        if len(functional) != 1 or len(distinct) != 1:
            return None
        selected = functional
        status = "counterevidence"
    return {"status": status, "facts": selected[:limit]}


def node_ids(relation_connection: sqlite3.Connection, qids: Iterable[str]) -> dict[str, int]:
    qids = list(dict.fromkeys(qids))
    if not qids:
        return {}
    placeholders = ",".join("?" for _ in qids)
    return dict(relation_connection.execute(
        f"SELECT qid,id FROM node WHERE qid IN ({placeholders})", qids
    ))


def compatible_year(row: sqlite3.Row, years: set[int]) -> bool:
    if not years:
        return True
    point = row["point_year"]
    start = row["start_year"]
    end = row["end_year"]
    if point is not None:
        if predicate_name(row["predicate"]) == "award received":
            return any(abs(point - value) <= 1 for value in years)
        return point in years
    if start is not None or end is not None:
        return any((start is None or start <= year) and (end is None or year <= end) for year in years)
    return True


def query_facts(
    connection: sqlite3.Connection,
    *,
    subject_ids: Iterable[int] = (),
    object_ids: Iterable[int] = (),
    predicates: Iterable[str],
    years: set[int],
    limit: int = 64,
) -> list[sqlite3.Row]:
    subjects = list(dict.fromkeys(subject_ids))
    objects = list(dict.fromkeys(object_ids))
    predicate_ids = [PREDICATE_IDS[name] for name in predicates if name in PREDICATE_IDS]
    if not predicate_ids or not (subjects or objects):
        return []
    clauses = []
    params: list[int] = []
    if subjects:
        clauses.append(f"f.subject IN ({','.join('?' for _ in subjects)})")
        params.extend(subjects)
    if objects:
        clauses.append(f"f.object IN ({','.join('?' for _ in objects)})")
        params.extend(objects)
    params.extend(predicate_ids)
    params.append(limit)
    rows = connection.execute(f"""
        SELECT f.*, s.qid AS subject_qid, s.label AS subject_label,
               o.qid AS object_qid, o.label AS object_label,
               a.label AS applies_label, w.label AS work_label
        FROM fact AS f
        JOIN node AS s ON s.id=f.subject
        LEFT JOIN node AS o ON o.id=f.object
        LEFT JOIN node AS a ON a.id=f.applies_to
        LEFT JOIN node AS w ON w.id=f.for_work
        WHERE ({' OR '.join(clauses)})
          AND f.predicate IN ({','.join('?' for _ in predicate_ids)})
        LIMIT ?
    """, params).fetchall()
    return [row for row in rows if compatible_year(row, years)]


def render_fact(row: sqlite3.Row) -> str:
    value = row["object_label"] or row["literal"] or "unknown"
    qualifiers = []
    if row["point_year"] is not None:
        qualifiers.append(str(row["point_year"]))
    if row["start_year"] is not None or row["end_year"] is not None:
        qualifiers.append(f"{row['start_year'] or '?'}–{row['end_year'] or '?'}")
    if row["applies_label"]:
        qualifiers.append(f"applies to {row['applies_label']}")
    if row["work_label"]:
        qualifiers.append(f"for work {row['work_label']}")
    suffix = f" [{'; '.join(qualifiers)}]" if qualifiers else ""
    return f"{predicate_name(row['predicate'])}: {value}{suffix}"


def answer_matches(row: sqlite3.Row, answer_qids: set[str], answer_values: Iterable[str]) -> bool:
    if row["object_qid"] and row["object_qid"] in answer_qids:
        return True
    if row["for_work"] and row["work_label"]:
        if normalize_name(row["work_label"]) in {normalize_name(value) for value in answer_values}:
            return True
    values = {normalize_name(value) for value in answer_values}
    return normalize_name(row["object_label"] or row["literal"] or "") in values


def retrieve_office_overlap(
    connection: sqlite3.Connection,
    subject_ids: Iterable[int],
    answer_ids: set[int],
) -> dict[str, Any] | None:
    """Constrained person→office→jurisdiction→monarch overlap query."""
    positions = query_facts(
        connection, subject_ids=subject_ids, predicates=("position held",), years=set(), limit=64
    )
    executive = [row for row in positions if re.search(
        r"\b(?:prime minister|premier|head of government)\b", row["object_label"] or "", re.I
    )]
    jurisdiction_pid = PREDICATE_IDS.get("applies to jurisdiction")
    if not executive or jurisdiction_pid is None:
        return None
    candidates: list[tuple[sqlite3.Row, sqlite3.Row]] = []
    for held in executive:
        jurisdictions = connection.execute(
            "SELECT object FROM fact WHERE subject=? AND predicate=? AND object IS NOT NULL",
            (held["object"], jurisdiction_pid),
        ).fetchall()
        for jurisdiction in jurisdictions:
            offices = connection.execute("""
                SELECT f.subject,n.label FROM fact AS f JOIN node AS n ON n.id=f.subject
                WHERE f.predicate=? AND f.object=?
            """, (jurisdiction_pid, jurisdiction[0])).fetchall()
            for office_id, office_label in offices:
                if not re.search(r"\b(?:monarch|king|queen|head of state)\b", office_label, re.I):
                    continue
                holders = query_facts(
                    connection, object_ids=(office_id,), predicates=("position held",),
                    years=set(), limit=64,
                )
                candidates.extend(
                    (held, holder) for holder in holders if intervals_overlap(held, holder)
                )
    if not candidates:
        return None
    candidates.sort(key=lambda pair: (
        pair[1]["subject"] not in answer_ids, pair[1]["start_year"] is None
    ))
    held, monarch = candidates[0]
    status = "support" if monarch["subject"] in answer_ids else "counterevidence"
    return {
        "status": status,
        "facts": [
            f"position held: {held['object_label']} [{held['start_year'] or '?'}–{held['end_year'] or '?'}]",
            f"concurrent office holder: {monarch['subject_label']} — {monarch['object_label']} "
            f"[{monarch['start_year'] or '?'}–{monarch['end_year'] or '?'}]",
        ],
        "rows": [dict(held), dict(monarch)],
    }


def retrieve_relation_evidence(
    entity_connection: sqlite3.Connection,
    relation_connection: sqlite3.Connection,
    conversation: str,
    *,
    limit: int = 6,
    allow_inverse: bool = True,
) -> dict[str, Any]:
    """Return direct and inverse relation facts, plus a conservative match status."""
    relation_connection.row_factory = sqlite3.Row
    claim = extract_claim_query(conversation)
    subject_qids = linked_subject_qids(entity_connection, claim)
    answer_qids = linked_qids(entity_connection, claim.answer_values)
    ids = node_ids(relation_connection, [*subject_qids, *answer_qids])
    trusted_qids = [
        qid for qid in trusted_subject_qids(entity_connection, claim, subject_qids)
        if qid in ids
    ]
    trusted_subject_ids = {ids[qid] for qid in trusted_qids if qid in ids}
    years = {int(value) for value in YEAR_RE.findall(claim.question)}
    predicates = [name for name in claim.predicates if name in PREDICATE_IDS]

    if re.search(r"\breign of which monarch\b", claim.question, re.I):
        overlap = retrieve_office_overlap(
            relation_connection,
            [ids[qid] for qid in subject_qids if qid in ids],
            {ids[qid] for qid in answer_qids if qid in ids},
        )
        if overlap:
            return {
                "claim": asdict(claim), "subject_qids": subject_qids,
                "answer_qids": answer_qids, "status": overlap["status"],
                "facts": overlap["facts"], "fact_rows": overlap["rows"],
                "used_inverse": True, "used_two_hop": True, "abstain_reason": None,
            }

    direct = query_facts(
        relation_connection,
        subject_ids=[ids[qid] for qid in subject_qids if qid in ids],
        predicates=predicates,
        years=years,
    )
    inverse_names = tuple(dict.fromkeys(
        inverse for predicate in predicates for inverse in INVERSE_PREDICATES.get(predicate, ())
    )) if allow_inverse else ()
    inverse = query_facts(
        relation_connection,
        object_ids=[ids[qid] for qid in subject_qids if qid in ids],
        predicates=inverse_names,
        years=years,
    )
    # An inverse fact is relevant only when its subject is the proposed answer,
    # or when it supplies the correct competing value for a functional relation.
    answer_id_set = {ids[qid] for qid in answer_qids if qid in ids}
    inverse_answer = [row for row in inverse if row["subject"] in answer_id_set]
    selected_inverse = inverse_answer or inverse

    matches = [
        row for row in direct
        if row["subject"] in trusted_subject_ids
        and answer_matches(row, set(answer_qids), claim.answer_values)
    ]
    matches.extend(row for row in inverse_answer)
    if matches:
        matched_subjects = {row["subject"] for row in matches if row in direct}
        if matched_subjects:
            direct = [row for row in direct if row["subject"] in matched_subjects]
    elif subject_qids:
        # Counterfacts from incidental names in the question are a major false-
        # positive source. Without an answer match, only trust the first linked
        # grammatical subject; later spans remain answer-recovery fallbacks.
        direct = [row for row in direct if row["subject"] in trusted_subject_ids]
    evidence_rows = []
    seen_rows: set[tuple[Any, ...]] = set()
    for row in [*matches, *direct, *selected_inverse]:
        key = (row["subject"], row["predicate"], row["object"], row["literal"],
               row["start_year"], row["end_year"], row["point_year"], row["for_work"])
        if key not in seen_rows:
            seen_rows.add(key)
            evidence_rows.append(row)
    evidence_rows = evidence_rows[:limit]
    direct_predicates = {predicate_name(row["predicate"]) for row in direct}
    has_functional_counterfact = bool(
        direct and direct_predicates & FUNCTIONAL_PREDICATES and not matches
    )
    has_work_counterfact = bool(direct and any(row["for_work"] for row in direct) and not matches)
    if matches:
        status = "support"
    elif has_functional_counterfact or has_work_counterfact:
        status = "counterevidence"
    else:
        status = "uncertain"
    return {
        "claim": asdict(claim),
        "subject_qids": subject_qids,
        "answer_qids": answer_qids,
        "status": status,
        "facts": [render_fact(row) for row in evidence_rows],
        "fact_rows": [dict(row) for row in evidence_rows],
        "used_inverse": bool(selected_inverse),
        "used_two_hop": False,
        "abstain_reason": None if evidence_rows else "no_relation_fact",
    }
