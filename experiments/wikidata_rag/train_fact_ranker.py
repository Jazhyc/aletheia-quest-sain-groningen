#!/usr/bin/env python3
"""Train a tiny hashed fallback ranker for already-shipped Wikidata cards."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Iterable
import zlib

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier

from experiments.wikidata_rag.claim_retrieval import (
    content_tokens,
    extract_claim_query,
    normalize,
)
from experiments.wikidata_rag.evaluate_retrieval import score_record
from experiments.wikidata_rag.evaluate_retrieval import STOPWORDS, TOKEN_RE


FEATURES = 1 << 15


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bucket(question: str) -> int:
    return hashlib.sha1(normalize(question).encode()).digest()[0] % 5


def feature_index(name: str) -> int:
    return zlib.crc32(name.encode()) & (FEATURES - 1)


def bounded_tokens(text: str, limit: int = 24) -> list[str]:
    selected = []
    for raw in TOKEN_RE.findall(text.lower().replace("’", "'")):
        token = raw.replace("’", "'")
        if len(token) < 3 or token in STOPWORDS:
            continue
        selected.append(token)
        if len(selected) == limit:
            break
    return selected


def feature_dict(
    question: str,
    answer: str,
    routed_predicates: set[str],
    subject: str,
    fact: str,
    popularity: int,
) -> dict[int, float]:
    predicate, _, value = fact.partition(":")
    predicate = predicate.strip().lower()
    q_tokens = bounded_tokens(question)
    a_tokens = set(bounded_tokens(answer))
    subject_tokens = set(bounded_tokens(subject))
    value_tokens = set(bounded_tokens(value))
    features: dict[str, float] = {
        "bias": 1.0,
        f"predicate={predicate}": 1.0,
        f"wh={normalize(question).partition(' ')[0]}": 1.0,
        "rule_predicate": float(predicate in routed_predicates),
        "subject_in_question": float(normalize(subject) in normalize(question)),
        "value_in_answer": float(bool(normalize(value)) and normalize(value) in normalize(answer)),
        "subject_question_overlap": float(len(subject_tokens & set(q_tokens))),
        "value_answer_overlap": float(len(value_tokens & a_tokens)),
        "value_question_overlap": float(len(value_tokens & set(q_tokens))),
        "question_fact_overlap": float(len(set(q_tokens) & (subject_tokens | value_tokens))),
        "has_qualifier": float("[" in fact),
        "for_work": float("for work" in fact.lower()),
        "year_overlap": float(bool(
            {token for token in q_tokens if token.isdigit()}
            & {token for token in bounded_tokens(fact) if token.isdigit()}
        )),
        "log_popularity": math.log1p(max(0, popularity)) / 20.0,
        "fact_token_count": len(bounded_tokens(fact, 64)) / 32.0,
    }
    for token in q_tokens:
        features[f"q={token}"] = 1.0
        features[f"qp={predicate}|{token}"] = 1.0
    for left, right in zip(q_tokens, q_tokens[1:]):
        features[f"qpair={left}|{right}"] = 1.0
        features[f"qppair={predicate}|{left}|{right}"] = 1.0
    for token in a_tokens:
        features[f"ap={predicate}|{token}"] = 1.0
    hashed: dict[int, float] = {}
    for name, value_number in features.items():
        index = feature_index(name)
        hashed[index] = hashed.get(index, 0.0) + float(value_number)
    return hashed


def card_candidates(
    connection: sqlite3.Connection, qids: Iterable[str]
) -> list[tuple[str, str, int]]:
    """Return bounded subject-predicate groups rather than individual values."""
    qids = list(dict.fromkeys(qids))
    if not qids:
        return []
    placeholders = ",".join("?" for _ in qids)
    rows = connection.execute(
        f"SELECT label,facts,popularity FROM entity WHERE qid IN ({placeholders})", qids
    ).fetchall()
    candidates = []
    for label, raw_facts, popularity in rows:
        grouped: dict[str, list[str]] = {}
        for fact in raw_facts.split("; "):
            predicate, separator, value = fact.partition(":")
            if separator:
                grouped.setdefault(predicate.strip(), []).append(value.strip())
        for predicate, values in grouped.items():
            candidates.append((
                label, f"{predicate}: {' | '.join(values[:8])}", int(popularity)
            ))
    return candidates


def matrix(rows: list[dict[int, float]]) -> csr_matrix:
    indices: list[int] = []
    data: list[float] = []
    indptr = [0]
    for row in rows:
        for index, value in sorted(row.items()):
            indices.append(index)
            data.append(value)
        indptr.append(len(indices))
    return csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int64)),
        shape=(len(rows), FEATURES),
    )


def row_selection(
    scores: np.ndarray,
    row_slices: list[tuple[int, int]],
    row_ids: Iterable[int],
    threshold: float,
    labels: np.ndarray,
    metrics: list[dict[str, float]],
    covered: list[bool],
) -> dict[str, float]:
    emitted = positive = novel_hits = 0
    precision_values = []
    for row_id in row_ids:
        if covered[row_id]:
            continue
        start, end = row_slices[row_id]
        if start == end:
            continue
        local = start + int(np.argmax(scores[start:end]))
        if scores[local] < threshold:
            continue
        emitted += 1
        positive += int(labels[local])
        novel_hits += int(metrics[local]["novel_target_recall"] > 0)
        precision_values.append(metrics[local]["evidence_precision"])
    return {
        "emitted": emitted,
        "positive": positive,
        "binary_precision": positive / max(1, emitted),
        "novel_hit_rows": novel_hits,
        "mean_evidence_precision": statistics.fmean(precision_values) if precision_values else 0.0,
    }


def choose_threshold(
    scores: np.ndarray,
    row_slices: list[tuple[int, int]],
    calibration_rows: list[int],
    labels: np.ndarray,
    metrics: list[dict[str, float]],
    covered: list[bool],
) -> tuple[float, dict[str, float]]:
    maxima = [
        float(np.max(scores[start:end]))
        for row_id in calibration_rows
        for start, end in [row_slices[row_id]]
        if start < end and not covered[row_id]
    ]
    thresholds = sorted(set(np.quantile(maxima, np.linspace(0, 1, 101)))) if maxima else [0.0]
    candidates = []
    for threshold in thresholds:
        report = row_selection(
            scores, row_slices, calibration_rows, float(threshold), labels, metrics, covered
        )
        if report["emitted"] >= 5 and report["binary_precision"] >= 0.60:
            candidates.append((report["positive"], report["emitted"], float(threshold), report))
    if not candidates:
        threshold = max(maxima, default=0.0) + 1e-6
        return threshold, row_selection(
            scores, row_slices, calibration_rows, threshold, labels, metrics, covered
        )
    _, _, threshold, report = max(candidates)
    return threshold, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-cache", type=Path, required=True)
    parser.add_argument("--retrieval-cache", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--entity-database", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    oracle_rows = load(args.oracle_cache)
    retrieval = {(row["dataset"], int(row["index"])): row for row in load(args.retrieval_cache)}
    teachers = {
        (row["dataset"], int(row["index"])): row for row in load(args.teacher_cache)
        if row.get("reasoning_summary") and not row.get("parse_error")
    }
    entity = sqlite3.connect(f"file:{args.entity_database.resolve()}?mode=ro", uri=True)
    features: list[dict[int, float]] = []
    labels_list: list[int] = []
    metrics: list[dict[str, float]] = []
    row_slices = []
    row_buckets = []
    covered = []
    for number, row in enumerate(oracle_rows, 1):
        key = (row["dataset"], int(row["index"]))
        teacher = teachers[key]
        retrieval_row = retrieval[key]
        claim = extract_claim_query(retrieval_row["conversation"])
        source = {
            "conversation": teacher["student_prompt"],
            "reasoning_summary": teacher["reasoning_summary"],
        }
        start = len(features)
        for subject, fact, popularity in card_candidates(entity, row["broad_qids"]):
            # Score only the value payload. Entity and predicate titles are
            # available to the ranker but must not manufacture oracle overlap.
            evidence = fact.partition(":")[2].strip()
            metric = score_record(source, evidence)
            features.append(feature_dict(
                claim.question, claim.answer, set(claim.predicates),
                subject, fact, popularity,
            ))
            metrics.append(metric)
            labels_list.append(int(
                metric["novel_target_recall"] > 0 and metric["evidence_precision"] >= 0.25
            ))
        row_slices.append((start, len(features)))
        row_buckets.append(bucket(claim.question))
        covered.append(bool(retrieval_row.get("real_passages")))
        if number % 250 == 0:
            print(f"features {number}/{len(oracle_rows)} candidates={len(features)}", flush=True)
    entity.close()

    x = matrix(features)
    y = np.asarray(labels_list, dtype=np.int8)
    train_rows = [index for index, value in enumerate(row_buckets) if value not in {0, 1}]
    calibration_rows = [index for index, value in enumerate(row_buckets) if value == 1]
    test_rows = [index for index, value in enumerate(row_buckets) if value == 0]
    train_candidates = np.concatenate([
        np.arange(*row_slices[row_id], dtype=np.int64) for row_id in train_rows
    ])
    reports = []
    for alpha in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4):
        model = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=alpha, class_weight="balanced",
            max_iter=50, tol=1e-4, random_state=42, average=True,
        )
        model.fit(x[train_candidates], y[train_candidates])
        scores = model.decision_function(x)
        threshold, calibration = choose_threshold(
            scores, row_slices, calibration_rows, y, metrics, covered
        )
        test = row_selection(
            scores, row_slices, test_rows, threshold, y, metrics, covered
        )
        reports.append({
            "alpha": alpha, "threshold": threshold,
            "calibration": calibration, "test": test, "model": model,
        })
    selected = max(
        reports,
        key=lambda row: (
            row["calibration"]["positive"],
            row["calibration"]["binary_precision"],
            -row["calibration"]["emitted"], -row["alpha"],
        ),
    )
    final_train_rows = train_rows + calibration_rows
    final_candidates = np.concatenate([
        np.arange(*row_slices[row_id], dtype=np.int64) for row_id in final_train_rows
    ])
    final_model = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=selected["alpha"], class_weight="balanced",
        max_iter=50, tol=1e-4, random_state=42, average=True,
    )
    final_model.fit(x[final_candidates], y[final_candidates])
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_model,
        weights=final_model.coef_.astype(np.float16),
        intercept=final_model.intercept_.astype(np.float32),
        threshold=np.asarray([selected["threshold"]], dtype=np.float32),
        features=np.asarray([FEATURES], dtype=np.int32),
    )
    report = {
        "rows": len(row_slices), "candidates": len(features),
        "positive_candidates": int(y.sum()), "group_buckets": Counter(row_buckets),
        "train_rows": len(train_rows), "calibration_rows": len(calibration_rows),
        "test_rows": len(test_rows),
        "candidates_by_split": {
            "train": int(len(train_candidates)), "final_train": int(len(final_candidates)),
        },
        "sweep": [{key: value for key, value in row.items() if key != "model"} for row in reports],
        "selected": {key: value for key, value in selected.items() if key != "model"},
        "model_bytes": args.output_model.stat().st_size,
    }
    args.output_report.write_text(json.dumps(report, indent=2, default=dict) + "\n")
    print(json.dumps(report, indent=2, default=dict))


if __name__ == "__main__":
    main()
