#!/usr/bin/env python3
"""Train a compact ranker from grounded GPT-OSS claim-fact supervision."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.wikidata_rag.train_fact_ranker import (
    bounded_tokens,
    feature_dict,
    feature_index,
    matrix,
)
from experiments.wikidata_rag.claim_retrieval import normalize


DECISIVE = {"supports", "contradicts", "decisive"}


def supervision_feature_dict(
    question: str,
    answer: str,
    routed_predicates: set[str],
    subject: str,
    fact: str,
    popularity: int,
) -> dict[int, float]:
    """Add compact relation-phrase interactions to the generic fact features."""
    features = feature_dict(
        question, answer, routed_predicates, subject, fact, popularity
    )
    predicate = fact.partition(":")[0].strip().lower()
    question_tokens = bounded_tokens(question, 32)
    names: dict[str, float] = {
        f"route_count={min(len(routed_predicates), 4)}": 1.0,
    }
    for route in sorted(routed_predicates):
        names[f"route_pair={route}|{predicate}"] = 1.0
    for left, middle, right in zip(
        question_tokens, question_tokens[1:], question_tokens[2:]
    ):
        names[f"qptriple={predicate}|{left}|{middle}|{right}"] = 1.0
    for name, value in names.items():
        index = feature_index(name)
        features[index] = features.get(index, 0.0) + value
    return features


def char_relation_feature_dict(
    question: str,
    answer: str,
    routed_predicates: set[str],
    subject: str,
    fact: str,
    popularity: int,
) -> dict[int, float]:
    """Learn wording-to-relation mappings while masking the candidate subject."""
    features = supervision_feature_dict(
        question, answer, routed_predicates, subject, fact, popularity
    )
    predicate = fact.partition(":")[0].strip().lower()
    text = f"  {normalize(question)[:400]}  "
    subject_text = normalize(subject)
    if subject_text:
        text = text.replace(subject_text, " subject ")
    for size in (3, 4, 5):
        for start in range(len(text) - size + 1):
            name = f"qc{size}={predicate}|{text[start:start + size]}"
            index = feature_index(name)
            features[index] = features.get(index, 0.0) + 1.0
    return features


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bucket(group: str) -> int:
    return hashlib.sha1(group.encode()).digest()[0] % 5


def safe_metric(function: Any, labels: np.ndarray, scores: np.ndarray) -> float:
    return float(function(labels, scores)) if len(set(labels.tolist())) > 1 else 0.0


def row_operating_point(
    scores: np.ndarray,
    labels: np.ndarray,
    row_slices: list[tuple[int, int]],
    row_ids: list[int],
    threshold: float,
    confidence_mode: str = "absolute",
) -> dict[str, float]:
    emitted = decisive = rows_with_decisive = top1_hits = 0
    for row_id in row_ids:
        start, end = row_slices[row_id]
        if start == end:
            continue
        row_scores = scores[start:end]
        local_offset = int(np.argmax(row_scores))
        local = start + local_offset
        if confidence_mode == "margin" and len(row_scores) > 1:
            confidence = float(row_scores[local_offset] - np.partition(row_scores, -2)[-2])
        else:
            confidence = float(row_scores[local_offset])
        has_decisive = bool(labels[start:end].any())
        rows_with_decisive += int(has_decisive)
        top1_hits += int(labels[local])
        if confidence >= threshold:
            emitted += 1
            decisive += int(labels[local])
    return {
        "rows": len(row_ids), "rows_with_decisive": rows_with_decisive,
        "top1_decisive": top1_hits,
        "emitted": emitted, "decisive": decisive,
        "precision": decisive / max(1, emitted),
        "recall_of_decisive_rows": decisive / max(1, rows_with_decisive),
    }


def choose_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    row_slices: list[tuple[int, int]],
    row_ids: list[int],
    minimum_precision: float,
    confidence_mode: str = "absolute",
) -> tuple[float, dict[str, float]]:
    maxima = []
    for row_id in row_ids:
        start, end = row_slices[row_id]
        row_scores = scores[start:end]
        if confidence_mode == "margin" and len(row_scores) > 1:
            top_two = np.partition(row_scores, -2)[-2:]
            maxima.append(float(top_two[1] - top_two[0]))
        elif len(row_scores):
            maxima.append(float(np.max(row_scores)))
    candidates = []
    for threshold in sorted(set(np.quantile(maxima, np.linspace(0, 1, 101)))):
        result = row_operating_point(
            scores, labels, row_slices, row_ids, float(threshold), confidence_mode
        )
        if result["emitted"] >= 5 and result["precision"] >= minimum_precision:
            candidates.append((result["decisive"], result["emitted"], float(threshold), result))
    if candidates:
        _, _, threshold, result = max(candidates)
        return threshold, result
    threshold = max(maxima, default=0.0) + 1e-6
    return threshold, row_operating_point(
        scores, labels, row_slices, row_ids, threshold, confidence_mode
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-input", type=Path)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.8)
    parser.add_argument(
        "--feature-mode", choices=("generic", "relation", "char_relation"),
        default="generic",
    )
    args = parser.parse_args()

    training_rows = [
        row for path in args.input for row in load(path)
        if not row.get("parse_error")
    ]
    validation_rows = [
        row for row in load(args.validation_input) if not row.get("parse_error")
    ] if args.validation_input else []
    rows = training_rows + validation_rows
    features = []
    labels_list = []
    row_slices = []
    groups = []
    synthetic_rows = []
    frozen_rows_flags = []
    label_counts: Counter[str] = Counter()
    feature_function = {
        "generic": feature_dict,
        "relation": supervision_feature_dict,
        "char_relation": char_relation_feature_dict,
    }[args.feature_mode]
    for row in rows:
        labels_by_id = {item["id"]: item for item in row["labels"]}
        start = len(features)
        for candidate in row["candidates"]:
            supervised = labels_by_id.get(candidate["id"])
            if supervised is None:
                continue
            label_counts[supervised["label"]] += 1
            features.append(feature_function(
                row["question"], row["answer_full"], set(row["rule_predicates"]),
                candidate["subject"], candidate["fact"], candidate["popularity"],
            ))
            labels_list.append(int(supervised["label"] in DECISIVE))
        row_slices.append((start, len(features)))
        groups.append(row["question_group"])
        synthetic_rows.append(bool(row.get("synthetic")))
        frozen_rows_flags.append(len(row_slices) > len(training_rows))

    x = matrix(features)
    y = np.asarray(labels_list, dtype=np.int8)
    buckets = np.asarray([bucket(group) for group in groups])
    development_ids = [index for index, value in enumerate(frozen_rows_flags) if not value]
    frozen_rows = [index for index, value in enumerate(frozen_rows_flags) if value]
    has_real = any(not synthetic_rows[index] for index in development_ids)
    has_synthetic = any(synthetic_rows[index] for index in development_ids)
    if has_real and has_synthetic:
        train_rows = [
            index for index in development_ids
            for value, split in [(synthetic_rows[index], buckets[index])]
            if value or split not in {0, 1}
        ]
        calibration_rows = [
            index for index in development_ids
            for value, split in [(synthetic_rows[index], buckets[index])]
            if not value and split == 1
        ]
        test_rows = [
            index for index in development_ids
            for value, split in [(synthetic_rows[index], buckets[index])]
            if not value and split == 0
        ]
    else:
        train_rows = [index for index in development_ids if buckets[index] not in {0, 1}]
        calibration_rows = [index for index in development_ids if buckets[index] == 1]
        test_rows = [index for index in development_ids if buckets[index] == 0]
    indices = lambda ids: np.concatenate([np.arange(*row_slices[row]) for row in ids]).astype(int)
    train_candidates = indices(train_rows)
    calibration_candidates = indices(calibration_rows)
    test_candidates = indices(test_rows)

    sweep = []
    models = []
    for class_weight in (None, "balanced"):
        for alpha in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4):
            model = SGDClassifier(
                loss="log_loss", penalty="l2", alpha=alpha,
                class_weight=class_weight, max_iter=200, tol=1e-5,
                random_state=42, average=True,
            )
            model.fit(x[train_candidates], y[train_candidates])
            scores = model.decision_function(x)
            for confidence_mode in ("absolute", "margin"):
                threshold, calibration = choose_threshold(
                    scores, y, row_slices, calibration_rows, args.minimum_precision,
                    confidence_mode,
                )
                test = row_operating_point(
                    scores, y, row_slices, test_rows, threshold, confidence_mode
                )
                sweep.append({
                    "alpha": alpha, "class_weight": class_weight,
                    "threshold": threshold, "confidence_mode": confidence_mode,
                    "calibration_candidate_auroc": safe_metric(
                        roc_auc_score, y[calibration_candidates], scores[calibration_candidates]
                    ),
                    "calibration_candidate_ap": safe_metric(
                        average_precision_score, y[calibration_candidates], scores[calibration_candidates]
                    ),
                    "test_candidate_auroc": safe_metric(
                        roc_auc_score, y[test_candidates], scores[test_candidates]
                    ),
                    "test_candidate_ap": safe_metric(
                        average_precision_score, y[test_candidates], scores[test_candidates]
                    ),
                    "calibration": calibration, "test": test,
                })
                models.append(model)
    selected_index = max(range(len(sweep)), key=lambda index: (
        sweep[index]["calibration"]["decisive"],
        sweep[index]["calibration"]["precision"],
        sweep[index]["calibration_candidate_ap"],
    ))
    selected = sweep[selected_index]
    # Preserve the score scale on which the threshold was calibrated. A refit
    # that includes calibration rows changes SGD margins and silently invalidates
    # the frozen operating point even when the ranking barely changes.
    final_model = models[selected_index]
    final_scores = final_model.decision_function(x)
    frozen_candidates = indices(frozen_rows) if frozen_rows else np.asarray([], dtype=int)
    frozen_validation = None
    if frozen_rows:
        training_groups = {groups[index] for index in development_ids}

        def frozen_report(row_ids: list[int]) -> dict[str, Any]:
            candidate_ids = indices(row_ids) if row_ids else np.asarray([], dtype=int)
            return {
                "candidate_auroc": safe_metric(
                    roc_auc_score, y[candidate_ids], final_scores[candidate_ids]
                ) if len(candidate_ids) else 0.0,
                "candidate_ap": safe_metric(
                    average_precision_score, y[candidate_ids], final_scores[candidate_ids]
                ) if len(candidate_ids) else 0.0,
                "operating_point": row_operating_point(
                    final_scores, y, row_slices, row_ids, selected["threshold"],
                    selected["confidence_mode"],
                ),
            }

        novel_rows = [index for index in frozen_rows if groups[index] not in training_groups]
        seen_rows = [index for index in frozen_rows if groups[index] in training_groups]
        frozen_validation = frozen_report(frozen_rows) | {
            "novel_question_groups": frozen_report(novel_rows),
            "seen_question_groups": frozen_report(seen_rows),
        }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_model, weights=final_model.coef_.astype(np.float16),
        intercept=final_model.intercept_.astype(np.float32),
        threshold=np.asarray([selected["threshold"]], dtype=np.float32),
        feature_mode=np.asarray([args.feature_mode]),
        confidence_mode=np.asarray([selected["confidence_mode"]]),
    )
    report = {
        "rows": len(rows), "candidates": len(y), "decisive_candidates": int(y.sum()),
        "feature_mode": args.feature_mode,
        "synthetic_rows": sum(synthetic_rows), "real_rows": len(rows) - sum(synthetic_rows),
        "development_rows": len(development_ids), "frozen_validation_rows": len(frozen_rows),
        "label_counts": label_counts, "split_rows": {
            "train": len(train_rows), "calibration": len(calibration_rows), "test": len(test_rows),
        },
        "sweep": sweep, "selected": selected, "frozen_validation": frozen_validation,
        "model_bytes": args.output_model.stat().st_size,
    }
    args.output_report.write_text(json.dumps(report, indent=2, default=dict) + "\n")
    print(json.dumps(report, indent=2, default=dict))


if __name__ == "__main__":
    main()
