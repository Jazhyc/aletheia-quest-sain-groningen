#!/usr/bin/env python3
"""Train a tiny grouped-holdout question-to-predicate router."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zlib

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier

from experiments.wikidata_rag.claim_retrieval import extract_claim_query, normalize


FEATURES = 1 << 13
WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def group_bucket(question: str) -> int:
    return hashlib.sha1(normalize(question).encode()).digest()[0] % 5


def oracle_predicate(fact: str) -> str:
    # Oracle caches intentionally omit entity titles, so a card fact is
    # rendered directly as ``predicate: value``.
    return fact.partition(":")[0].strip().lower()


def question_features(question: str) -> dict[int, float]:
    text = " ".join(question.lower().split())[:400]
    words = WORD_RE.findall(text)
    names: dict[str, float] = {"bias": 1.0}
    for word in words:
        names[f"w={word}"] = 1.0
    for size in (2, 3):
        for start in range(len(words) - size + 1):
            names[f"wg{size}={'|'.join(words[start:start + size])}"] = 1.0
    padded = f"  {text}  "
    for size in (3, 4, 5):
        for start in range(len(padded) - size + 1):
            names[f"c{size}={padded[start:start + size]}"] = 1.0
    hashed: dict[int, float] = {}
    for name, value in names.items():
        index = zlib.crc32(name.encode()) & (FEATURES - 1)
        hashed[index] = hashed.get(index, 0.0) + value
    return hashed


def matrix(rows: list[dict[int, float]]) -> csr_matrix:
    indices = []
    data = []
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


def predictions(scores: np.ndarray, model_classes: np.ndarray, top_k: int) -> np.ndarray:
    top_k = min(top_k, scores.shape[1])
    columns = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
    return model_classes[columns]


def accuracy(
    scores: np.ndarray, labels: np.ndarray, model_classes: np.ndarray, top_k: int
) -> float:
    predicted = predictions(scores, model_classes, top_k)
    return float(np.mean([label in row for label, row in zip(labels, predicted)]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-cache", type=Path, required=True)
    parser.add_argument("--retrieval-cache", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    retrieval = {(row["dataset"], int(row["index"])): row for row in load(args.retrieval_cache)}
    rows = []
    for oracle in load(args.oracle_cache):
        if not oracle["full_cards"]["high_precision_hit"]:
            continue
        key = (oracle["dataset"], int(oracle["index"]))
        record = retrieval[key]
        claim = extract_claim_query(record["conversation"])
        rows.append({
            "question": claim.question,
            "predicate": oracle_predicate(oracle["full_cards"]["fact"]),
            "covered": bool(record.get("real_passages")),
            "rule_predicates": set(claim.predicates),
        })
    classes = sorted({row["predicate"] for row in rows})
    class_ids = {value: index for index, value in enumerate(classes)}
    x = matrix([question_features(row["question"]) for row in rows])
    y = np.asarray([class_ids[row["predicate"]] for row in rows], dtype=np.int32)
    buckets = np.asarray([group_bucket(row["question"]) for row in rows])
    train = np.flatnonzero(~np.isin(buckets, [0, 1]))
    calibration = np.flatnonzero(buckets == 1)
    test = np.flatnonzero(buckets == 0)
    sweep = []
    models = []
    for alpha in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4):
        model = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=alpha, class_weight="balanced",
            max_iter=100, tol=1e-4, random_state=42, average=True,
        )
        model.fit(x[train], y[train])
        calibration_scores = model.decision_function(x[calibration])
        test_scores = model.decision_function(x[test])
        report = {
            "alpha": alpha,
            "calibration_top1": accuracy(calibration_scores, y[calibration], model.classes_, 1),
            "calibration_top3": accuracy(calibration_scores, y[calibration], model.classes_, 3),
            "test_top1": accuracy(test_scores, y[test], model.classes_, 1),
            "test_top3": accuracy(test_scores, y[test], model.classes_, 3),
        }
        sweep.append(report)
        models.append(model)
    selected_index = max(
        range(len(sweep)),
        key=lambda index: (
            sweep[index]["calibration_top3"], sweep[index]["calibration_top1"],
            -sweep[index]["alpha"],
        ),
    )
    selected = sweep[selected_index]
    final_train = np.flatnonzero(buckets != 0)
    final_model = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=selected["alpha"], class_weight="balanced",
        max_iter=100, tol=1e-4, random_state=42, average=True,
    )
    final_model.fit(x[final_train], y[final_train])
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_model,
        weights=final_model.coef_.astype(np.float16),
        intercept=final_model.intercept_.astype(np.float32),
        classes=np.asarray([classes[index] for index in final_model.classes_]),
        features=np.asarray([FEATURES], dtype=np.int32),
    )
    test_rows = [rows[index] for index in test]
    test_labels = y[test]
    test_scores = models[selected_index].decision_function(x[test])
    top3 = predictions(test_scores, models[selected_index].classes_, 3)
    uncovered = [index for index, row in enumerate(test_rows) if not row["covered"]]
    rule_hits = sum(
        row["predicate"] in row["rule_predicates"] for row in test_rows if not row["covered"]
    )
    learned_hits = sum(test_labels[index] in top3[index] for index in uncovered)
    report = {
        "rows": len(rows), "classes": len(classes),
        "class_counts": Counter(row["predicate"] for row in rows),
        "train_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
        "sweep": sweep, "selected": selected,
        "uncovered_test_oracle_rows": len(uncovered),
        "uncovered_rule_predicate_hits": rule_hits,
        "uncovered_learned_top3_hits": learned_hits,
        "model_bytes": args.output_model.stat().st_size,
    }
    args.output_report.write_text(json.dumps(report, indent=2, default=dict) + "\n")
    print(json.dumps(report, indent=2, default=dict))


if __name__ == "__main__":
    main()
