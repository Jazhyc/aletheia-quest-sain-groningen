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
    entity_context_score,
    extract_claim_query,
    normalize_name,
    query_candidate_entities,
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
    first_subject_qids = linked_qids(entity_connection, claim.subjects[:1])
    answer_qids = linked_qids(entity_connection, claim.answer_values)
    ids = node_ids(relation_connection, [*subject_qids, *answer_qids])
    trusted_qids = [qid for qid in first_subject_qids if qid in ids]
    coordinated = len(claim.subjects) > 1 and bool(re.search(r"\band\b", claim.question, re.I))
    typed_focus = any(pattern.search(claim.question) for pattern, _ in TYPE_HINTS)
    if not trusted_qids and subject_qids and (coordinated or typed_focus):
        trusted_qids = [subject_qids[0]]
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
