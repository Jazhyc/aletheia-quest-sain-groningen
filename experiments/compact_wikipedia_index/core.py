"""Build and query a compact SQLite FTS index of Wikipedia sentences."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
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
FEATURE_NAMES = (
    "claim_lexical",
    "question_lexical",
    "claim_title",
    "question_title",
    "claim_number_recall",
    "question_number_recall",
    "claim_rrf",
    "question_rrf",
    "combined_rrf",
    "sentence_lead",
    "length_similarity",
)


@dataclass(frozen=True)
class LinearReranker:
    """Small standardized logistic reranker with no sklearn runtime dependency."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    threshold: float

    @classmethod
    def from_json(cls, path: Path | str) -> "LinearReranker":
        payload = json.loads(Path(path).read_text())
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError("reranker feature contract does not match runtime")
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),
            scale=tuple(float(value) for value in payload["scale"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            intercept=float(payload["intercept"]),
            threshold=float(payload["threshold"]),
        )

    def probability(self, features: Iterable[float]) -> float:
        values = tuple(float(value) for value in features)
        if not (
            len(values)
            == len(self.mean)
            == len(self.scale)
            == len(self.coefficients)
        ):
            raise ValueError("reranker feature length mismatch")
        logit = self.intercept + sum(
            coefficient * (value - mean) / (scale or 1.0)
            for value, mean, scale, coefficient in zip(
                values, self.mean, self.scale, self.coefficients, strict=True
            )
        )
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)


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


def _reciprocal_rank(rank: int | None) -> float:
    return 1.0 / (2.0 + rank) if rank is not None else 0.0


def _length_similarity(left: str, right: str) -> float:
    left_length = max(1, len(TOKEN.findall(left)))
    right_length = max(1, len(TOKEN.findall(right)))
    return min(left_length, right_length) / max(left_length, right_length)


def candidate_features(
    question: str,
    claim: str,
    candidate: dict[str, Any],
    *,
    claim_rank: int | None,
    question_rank: int | None,
    combined_rank: int | None,
) -> tuple[float, ...]:
    """Compute proposition-aware features used by the frozen linear reranker."""
    text = str(candidate["text"])
    title = str(candidate["title"])
    claim_numbers = _number_tokens(claim)
    question_numbers = _number_tokens(question)
    text_numbers = _number_tokens(text)

    def number_recall(numbers: set[str]) -> float:
        return len(numbers & text_numbers) / len(numbers) if numbers else 1.0

    features = (
        lexical_relevance(claim, text),
        lexical_relevance(question, text),
        _title_overlap(claim, title),
        _title_overlap(question, title),
        number_recall(claim_numbers),
        number_recall(question_numbers),
        _reciprocal_rank(claim_rank),
        _reciprocal_rank(question_rank),
        _reciprocal_rank(combined_rank),
        1.0 / (1.0 + int(candidate["sentence_index"])),
        _length_similarity(claim, text),
    )
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError("reranker feature contract is inconsistent")
    return features


def retrieve_proposition(
    connection: sqlite3.Connection,
    question: str,
    claim: str,
    *,
    limit: int = 5,
    candidate_limit: int = 30,
    reranker: LinearReranker | None = None,
) -> list[dict[str, Any]]:
    """Union three FTS views and rerank candidates around an atomic proposition."""
    query_views = (claim, question, f"{claim} {question}")
    pools = [
        retrieve(
            connection,
            query,
            limit=candidate_limit,
            candidate_limit=candidate_limit,
        )
        for query in query_views
    ]
    by_id: dict[int, dict[str, Any]] = {}
    ranks: list[dict[int, int]] = []
    for pool in pools:
        rank_map: dict[int, int] = {}
        for rank, candidate in enumerate(pool):
            passage_id = int(candidate["passage_id"])
            rank_map[passage_id] = rank
            by_id.setdefault(passage_id, candidate)
        ranks.append(rank_map)

    output = []
    for passage_id, candidate in by_id.items():
        features = candidate_features(
            question,
            claim,
            candidate,
            claim_rank=ranks[0].get(passage_id),
            question_rank=ranks[1].get(passage_id),
            combined_rank=ranks[2].get(passage_id),
        )
        score = (
            reranker.probability(features)
            if reranker is not None
            else (
                features[0]
                + 0.20 * features[1]
                + 0.20 * features[2]
                + 0.10 * features[3]
                + 0.20 * features[4]
                + 0.10 * features[6]
                + 0.05 * features[7]
                + 0.05 * features[8]
            )
        )
        output.append({
            **candidate,
            "feature_names": FEATURE_NAMES,
            "features": features,
            "reranker_score": score,
        })
    output.sort(
        key=lambda item: (
            float(item["reranker_score"]),
            float(item["retrieval_score"]),
            -int(item["sentence_index"]),
        ),
        reverse=True,
    )
    return output[:limit]


def adjacent_window(
    connection: sqlite3.Connection,
    candidate: dict[str, Any],
    *,
    radius: int = 1,
    maximum_chars: int = 1200,
) -> str:
    """Return a bounded page-local window around a retrieved sentence."""
    rows = connection.execute(
        """
        SELECT text FROM passages
        WHERE title = ? AND sentence_index BETWEEN ? AND ?
        ORDER BY sentence_index
        """,
        (
            str(candidate["title"]),
            int(candidate["sentence_index"]) - radius,
            int(candidate["sentence_index"]) + radius,
        ),
    ).fetchall()
    window = normalize_text(" ".join(str(row[0]) for row in rows))
    return window[:maximum_chars].rsplit(" ", 1)[0] if len(window) > maximum_chars else window


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
