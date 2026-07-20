"""Build and query a compact SQLite FTS index of Wikipedia sentences."""

from __future__ import annotations

from collections.abc import Iterable
import re
import sqlite3
from typing import Any

from experiments.fever_fact_verification.core import (
    STOPWORDS,
    TOKEN,
    lexical_relevance,
    normalize_text,
    split_sentences,
)


SOURCE_MARKER = "Source sentence:"
FTS_RESERVED = {"and", "or", "not", "near"}


def query_terms(query: str, *, limit: int = 24) -> list[str]:
    """Return stable content terms suitable for a permissive FTS OR query."""
    terms: list[str] = []
    seen: set[str] = set()
    for raw in TOKEN.findall(query):
        term = raw.casefold()
        if (
            len(term) < 2
            or term in STOPWORDS
            or term in FTS_RESERVED
            or term in seen
        ):
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def fts_expression(query: str, *, limit: int = 24) -> str:
    """Build an escaped disjunction; candidate precision is handled downstream."""
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in query_terms(query, limit=limit))


def iter_page_sentences(
    pages: Iterable[dict[str, Any]],
    *,
    minimum_chars: int = 32,
    maximum_chars: int = 900,
) -> Iterable[dict[str, Any]]:
    """Yield deduplicated factual sentences with page-local offsets."""
    seen: set[tuple[str, str]] = set()
    for page in pages:
        if page.get("missing") or page.get("error"):
            continue
        title = normalize_text(str(
            page.get("canonical_title") or page.get("requested_title") or ""
        ))
        url = str(page.get("url") or "")
        if not title:
            continue
        for sentence_index, sentence in enumerate(
            split_sentences(str(page.get("extract") or ""), minimum_chars=minimum_chars)
        ):
            sentence = normalize_text(sentence)
            if not sentence or len(sentence) > maximum_chars:
                continue
            key = (title.casefold(), sentence.casefold())
            if key in seen:
                continue
            seen.add(key)
            yield {
                "title": title,
                "url": url,
                "sentence_index": sentence_index,
                "text": sentence,
            }


def create_index(
    connection: sqlite3.Connection,
    passages: Iterable[dict[str, Any]],
) -> dict[str, int]:
    """Create an external-content FTS5 index and return corpus counts."""
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        DROP TABLE IF EXISTS passages_fts;
        DROP TABLE IF EXISTS passages;
        CREATE TABLE passages (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            sentence_index INTEGER NOT NULL,
            text TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE passages_fts USING fts5(
            title,
            text,
            content='passages',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )
    batch = [
        (
            str(item["title"]),
            str(item.get("url") or ""),
            int(item["sentence_index"]),
            str(item["text"]),
        )
        for item in passages
    ]
    connection.executemany(
        "INSERT INTO passages(title, url, sentence_index, text) VALUES (?, ?, ?, ?)",
        batch,
    )
    connection.execute("INSERT INTO passages_fts(passages_fts) VALUES ('rebuild')")
    connection.commit()
    pages = connection.execute(
        "SELECT COUNT(DISTINCT title) FROM passages"
    ).fetchone()[0]
    return {"pages": int(pages), "passages": len(batch)}


def _number_tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN.findall(value)
        if any(character.isdigit() for character in token)
    }


def _title_overlap(query: str, title: str) -> float:
    query_content = {
        token.casefold()
        for token in TOKEN.findall(query)
        if token.casefold() not in STOPWORDS
    }
    title_content = {
        token.casefold()
        for token in TOKEN.findall(title)
        if token.casefold() not in STOPWORDS
    }
    return len(query_content & title_content) / max(1, len(title_content))


def retrieve(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 5,
    candidate_limit: int = 100,
) -> list[dict[str, Any]]:
    """Retrieve with FTS, then rerank for entity and number-sensitive overlap."""
    expression = fts_expression(query)
    if not expression or limit < 1:
        return []
    rows = connection.execute(
        """
        SELECT p.id, p.title, p.url, p.sentence_index, p.text,
               bm25(passages_fts, 2.0, 1.0) AS bm25_score
        FROM passages_fts
        JOIN passages AS p ON p.id = passages_fts.rowid
        WHERE passages_fts MATCH ?
        ORDER BY bm25_score
        LIMIT ?
        """,
        (expression, max(limit, candidate_limit)),
    ).fetchall()
    query_numbers = _number_tokens(query)
    ranked = []
    for row in rows:
        text = str(row[4])
        text_numbers = _number_tokens(text)
        number_recall = (
            len(query_numbers & text_numbers) / len(query_numbers)
            if query_numbers
            else 1.0
        )
        lexical_score = lexical_relevance(query, text)
        title_score = _title_overlap(query, str(row[1]))
        retrieval_score = lexical_score + 0.20 * title_score + 0.20 * number_recall
        ranked.append({
            "passage_id": int(row[0]),
            "title": str(row[1]),
            "url": str(row[2]),
            "sentence_index": int(row[3]),
            "text": text,
            "bm25_score": float(row[5]),
            "lexical_score": lexical_score,
            "title_score": title_score,
            "number_recall": number_recall,
            "retrieval_score": retrieval_score,
        })
    ranked.sort(
        key=lambda item: (
            float(item["retrieval_score"]),
            -float(item["bm25_score"]),
            -int(item["sentence_index"]),
        ),
        reverse=True,
    )
    return ranked[:limit]


def reference_source(text: str) -> str:
    """Extract the cited source sentence from a reader-formatted reference."""
    _, marker, source = str(text).partition(SOURCE_MARKER)
    return normalize_text(source if marker else text)


def source_matches(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    """Match a sentence against a one- or two-sentence audited source span."""
    if normalize_text(str(candidate.get("title") or "")).casefold() != normalize_text(
        str(reference.get("title") or "")
    ).casefold():
        return False
    candidate_text = normalize_text(str(candidate.get("text") or "")).casefold()
    reference_text = reference_source(str(reference.get("text") or "")).casefold()
    return bool(candidate_text) and (
        candidate_text in reference_text or reference_text in candidate_text
    )
