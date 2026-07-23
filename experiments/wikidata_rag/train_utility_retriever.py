#!/usr/bin/env python3
"""Train a compact fact ranker with an explicit downstream-utility abstention action."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import macro_metrics
from experiments.wikidata_rag.claim_retrieval import normalize
from experiments.wikidata_rag.train_fact_ranker import feature_dict, feature_index, matrix


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def group_bucket(question_group: str) -> int:
    return hashlib.sha1(question_group.encode()).digest()[0] % 5


def safe_metric(function: Any, labels: np.ndarray, scores: np.ndarray) -> float:
    return float(function(labels, scores)) if len(set(labels.tolist())) > 1 else 0.0


def load_semantic_model(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    artifact = np.load(path, allow_pickle=False)
    return {
        "weights": artifact["weights"].astype(np.float32).reshape(-1),
        "intercept": float(artifact["intercept"].reshape(-1)[0]),
        "path": path.resolve().as_posix(),
    }


def semantic_scores(
    row: dict[str, Any], semantic_model: dict[str, Any] | None
) -> list[float]:
    if semantic_model is None:
        return [0.0] * len(row["candidates"])
    values = []
    for candidate in row["candidates"]:
        features = feature_dict(
            row["question"], row["answer_full"], set(row["rule_predicates"]),
            candidate["subject"], candidate["fact"], candidate["popularity"],
        )
        values.append(
            semantic_model["intercept"]
            + sum(semantic_model["weights"][index] * value for index, value in features.items())
        )
    return values


def utility_feature_dict(
    row: dict[str, Any],
    candidate: dict[str, Any],
    *,
    feature_mode: str,
    semantic_score: float,
    semantic_rank: int,
) -> dict[int, float]:
    features = feature_dict(
        row["question"], row["answer_full"], set(row["rule_predicates"]),
        candidate["subject"], candidate["fact"], candidate["popularity"],
    )
    if feature_mode == "generic":
        return features
    baseline = float(row["empty_score"])
    uncertainty = 1.0 - 2.0 * abs(baseline - 0.5)
    predicate = candidate["fact"].partition(":")[0].strip().lower()
    names = {
        "reader_baseline_score": baseline,
        "reader_uncertainty": uncertainty,
        "reader_prediction": float(baseline >= 0.5),
        f"reader_predicate={predicate}|prediction={int(baseline >= 0.5)}": 1.0,
        f"reader_predicate={predicate}|uncertainty": uncertainty,
    }
    if feature_mode == "reader_semantic":
        names.update({
            "semantic_score": max(-10.0, min(10.0, semantic_score)) / 10.0,
            "semantic_rank_reciprocal": 1.0 / (semantic_rank + 1),
            f"semantic_top={int(semantic_rank == 0)}": 1.0,
            f"semantic_rank_bin={min(semantic_rank, 4)}": 1.0,
        })
    for name, value in names.items():
        index = feature_index(name)
        features[index] = features.get(index, 0.0) + float(value)
    return features


def build_examples(
    rows: list[dict[str, Any]],
    feature_mode: str,
    semantic_model: dict[str, Any] | None,
) -> tuple[Any, dict[str, np.ndarray], list[tuple[int, int]]]:
    features: list[dict[int, float]] = []
    raw_targets: list[float] = []
    controlled_targets: list[float] = []
    binary_targets: list[float] = []
    controlled_binary_targets: list[float] = []
    row_slices = []
    for row in rows:
        scores = semantic_scores(row, semantic_model)
        ranks = np.argsort(np.argsort(-np.asarray(scores))).tolist() if scores else []
        start = len(features)
        for candidate, score, rank in zip(row["candidates"], scores, ranks, strict=True):
            features.append(utility_feature_dict(
                row, candidate, feature_mode=feature_mode,
                semantic_score=score, semantic_rank=int(rank),
            ))
            raw_targets.append(float(candidate["utility"]))
            controlled_targets.append(float(candidate["controlled_utility"]))
            baseline_correct = int((float(row["empty_score"]) >= 0.5) == bool(row["label"]))
            shuffled_correct = int(
                (float(row["shuffled_control"]["score"]) >= 0.5) == bool(row["label"])
            )
            candidate_correct = int(
                (float(candidate["score"]) >= 0.5) == bool(row["label"])
            )
            binary_targets.append(float(candidate_correct - baseline_correct))
            controlled_binary_targets.append(float(candidate_correct - shuffled_correct))
        row_slices.append((start, len(features)))
    return matrix(features), {
        "utility": np.asarray(raw_targets, dtype=np.float64),
        "controlled_utility": np.asarray(controlled_targets, dtype=np.float64),
        "binary_utility": np.asarray(binary_targets, dtype=np.float64),
        "controlled_binary_utility": np.asarray(
            controlled_binary_targets, dtype=np.float64
        ),
    }, row_slices


def candidate_indices(
    row_slices: list[tuple[int, int]], row_ids: Iterable[int]
) -> np.ndarray:
    parts = [np.arange(*row_slices[row_id], dtype=np.int64) for row_id in row_ids]
    return np.concatenate(parts) if parts else np.asarray([], dtype=np.int64)


def score_metrics(rows: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    frame = pd.DataFrame({
        "dataset": [row["dataset"] for row in rows],
        "index": [row["index"] for row in rows],
        "label": [row["label"] for row in rows],
        "score": scores,
    })
    return macro_metrics(frame, 0.5)


def selection_report(
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    row_slices: list[tuple[int, int]],
    row_ids: list[int],
    threshold: float,
    *,
    minimum_gain: float,
) -> dict[str, Any]:
    selected_scores = []
    shuffled_scores = []
    baseline_scores = []
    raw_gains = []
    controlled_gains = []
    selected_candidate_ids: list[str | None] = []
    positive = harmful = decisive = semantic_known = 0
    for row_id in row_ids:
        row = rows[row_id]
        baseline = float(row["empty_score"])
        start, end = row_slices[row_id]
        chosen = None
        if start < end:
            local = int(np.argmax(predictions[start:end]))
            if float(predictions[start + local]) >= threshold:
                chosen = row["candidates"][local]
        if chosen is None:
            selected_scores.append(baseline)
            shuffled_scores.append(baseline)
            raw_gains.append(0.0)
            controlled_gains.append(0.0)
            selected_candidate_ids.append(None)
            continue
        raw_gain = float(chosen["utility"])
        controlled_gain = float(chosen["controlled_utility"])
        selected_scores.append(float(chosen["score"]))
        shuffled_scores.append(float(row["shuffled_control"]["score"]))
        raw_gains.append(raw_gain)
        controlled_gains.append(controlled_gain)
        selected_candidate_ids.append(str(chosen["id"]))
        positive += int(controlled_gain > minimum_gain)
        harmful += int(raw_gain < -minimum_gain)
        if "semantic_label" in chosen:
            semantic_known += 1
            decisive += int(chosen["semantic_label"] == "decisive")
    selected = sum(value is not None for value in selected_candidate_ids)
    labels = np.asarray([rows[row_id]["label"] for row_id in row_ids])

    def balanced(values: list[float]) -> float:
        array = np.asarray(values)
        means = [float(array[labels == label].mean()) for label in (0, 1) if (labels == label).any()]
        return float(np.mean(means)) if means else 0.0

    subset = [rows[row_id] for row_id in row_ids]
    baseline_scores = [float(row["empty_score"]) for row in subset]
    baseline_metrics = score_metrics(subset, baseline_scores)
    selected_metrics = score_metrics(subset, selected_scores)
    shuffled_metrics = score_metrics(subset, shuffled_scores)
    baseline_ba = baseline_metrics.get("balanced_accuracy")
    selected_ba = selected_metrics.get("balanced_accuracy")
    baseline_auroc = baseline_metrics.get("auroc")
    selected_auroc = selected_metrics.get("auroc")
    return {
        "rows": len(row_ids),
        "emitted": selected,
        "coverage": selected / max(1, len(row_ids)),
        "controlled_positive": positive,
        "controlled_positive_precision": positive / max(1, selected),
        "raw_harmful": harmful,
        "raw_harm_rate": harmful / max(1, selected),
        "semantic_known": semantic_known,
        "semantic_decisive": decisive,
        "semantic_decisive_precision": decisive / max(1, semantic_known),
        "balanced_raw_gain": balanced(raw_gains),
        "balanced_controlled_gain": balanced(controlled_gains),
        "mean_raw_gain_emitted": (
            sum(raw_gains) / selected if selected else 0.0
        ),
        "mean_controlled_gain_emitted": (
            sum(controlled_gains) / selected if selected else 0.0
        ),
        "balanced_accuracy_delta": (
            float(selected_ba - baseline_ba)
            if selected_ba is not None and baseline_ba is not None else None
        ),
        "auroc_delta": (
            float(selected_auroc - baseline_auroc)
            if selected_auroc is not None and baseline_auroc is not None else None
        ),
        "baseline_metrics": baseline_metrics,
        "selected_metrics": selected_metrics,
        "matched_shuffled_metrics": shuffled_metrics,
        "selected_candidate_ids": selected_candidate_ids,
    }


def choose_threshold(
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    row_slices: list[tuple[int, int]],
    row_ids: list[int],
    *,
    minimum_gain: float,
    minimum_precision: float,
    minimum_emitted: int,
    selection_objective: str,
) -> tuple[float, dict[str, Any]]:
    def fast_statistics(threshold: float) -> dict[str, float | int | None]:
        selected_scores = []
        baseline_scores = []
        labels = []
        datasets = []
        raw_gains = []
        controlled_gains = []
        emitted = positive = 0
        for row_id in row_ids:
            row = rows[row_id]
            baseline = float(row["empty_score"])
            start, end = row_slices[row_id]
            chosen = None
            if start < end:
                local = int(np.argmax(predictions[start:end]))
                if float(predictions[start + local]) >= threshold:
                    chosen = row["candidates"][local]
            labels.append(int(row["label"]))
            datasets.append(str(row["dataset"]))
            baseline_scores.append(baseline)
            if chosen is None:
                selected_scores.append(baseline)
                raw_gains.append(0.0)
                controlled_gains.append(0.0)
            else:
                emitted += 1
                selected_scores.append(float(chosen["score"]))
                raw_gains.append(float(chosen["utility"]))
                controlled = float(chosen["controlled_utility"])
                controlled_gains.append(controlled)
                positive += int(controlled > minimum_gain)

        label_array = np.asarray(labels)

        def balanced_gain(values: list[float]) -> float:
            array = np.asarray(values)
            means = [
                float(array[label_array == label].mean())
                for label in (0, 1) if (label_array == label).any()
            ]
            return float(np.mean(means)) if means else 0.0

        def macro_ba(scores: list[float]) -> float | None:
            score_array = np.asarray(scores)
            values = []
            for dataset in sorted(set(datasets)):
                mask = np.asarray([value == dataset for value in datasets])
                group_labels = label_array[mask]
                if not (group_labels == 0).any() or not (group_labels == 1).any():
                    continue
                predictions_binary = score_array[mask] >= 0.5
                recall = float(predictions_binary[group_labels == 1].mean())
                fpr = float(predictions_binary[group_labels == 0].mean())
                values.append((recall + 1.0 - fpr) / 2.0)
            return float(np.mean(values)) if values else None

        baseline_ba = macro_ba(baseline_scores)
        selected_ba = macro_ba(selected_scores)
        return {
            "emitted": emitted,
            "coverage": emitted / max(1, len(row_ids)),
            "controlled_positive_precision": positive / max(1, emitted),
            "balanced_raw_gain": balanced_gain(raw_gains),
            "balanced_controlled_gain": balanced_gain(controlled_gains),
            "balanced_accuracy_delta": (
                selected_ba - baseline_ba
                if selected_ba is not None and baseline_ba is not None else None
            ),
        }

    maxima = [
        float(np.max(predictions[start:end]))
        for row_id in row_ids
        for start, end in [row_slices[row_id]] if start < end
    ]
    abstain_threshold = max(maxima, default=0.0) + 1e-6
    abstain = selection_report(
        rows, predictions, row_slices, row_ids, abstain_threshold,
        minimum_gain=minimum_gain,
    )
    candidates: list[tuple[float, float, float, float, float, dict[str, Any]]] = []
    quantiles = np.linspace(0.0, 1.0, min(201, max(2, len(maxima))))
    for threshold in sorted(set(float(value) for value in np.quantile(maxima, quantiles))):
        statistics = fast_statistics(threshold)
        objective = (
            statistics["balanced_controlled_gain"]
            if selection_objective == "controlled_gain"
            else statistics["balanced_accuracy_delta"]
        )
        if (
            statistics["emitted"] >= minimum_emitted
            and statistics["controlled_positive_precision"] >= minimum_precision
            and objective is not None and objective > 0.0
        ):
            candidates.append((
                float(objective),
                float(statistics["balanced_controlled_gain"]),
                float(statistics["balanced_raw_gain"]),
                -float(statistics["coverage"]), threshold, statistics,
            ))
    if not candidates:
        return abstain_threshold, abstain
    *_, threshold, _ = max(candidates)
    return float(threshold), selection_report(
        rows, predictions, row_slices, row_ids, float(threshold),
        minimum_gain=minimum_gain,
    )


def oracle_predictions(
    rows: list[dict[str, Any]], row_slices: list[tuple[int, int]], target: np.ndarray
) -> np.ndarray:
    values = np.full(len(target), -math.inf, dtype=np.float64)
    for row_id, (start, end) in enumerate(row_slices):
        if start == end:
            continue
        local = start + int(np.argmax(target[start:end]))
        if target[local] > 0.0:
            values[local] = 1.0
    return values


def fit_models(
    x: Any,
    targets: dict[str, np.ndarray],
    train_candidates: np.ndarray,
) -> list[tuple[dict[str, Any], Any, np.ndarray]]:
    fitted = []
    for target_name, target in targets.items():
        scale = max(float(np.std(target[train_candidates])), 1e-3)
        normalized = target / scale
        for alpha in (0.1, 1.0, 10.0, 100.0):
            model = Ridge(alpha=alpha, solver="lsqr")
            model.fit(x[train_candidates], normalized[train_candidates])
            predictions = model.predict(x) * scale
            fitted.append(({
                "estimator": "ridge", "target": target_name,
                "alpha": alpha, "target_scale": scale,
            }, model, predictions))
        gain_thresholds = (
            (0.0,) if "binary" in target_name else (0.0, 0.01, 0.03)
        )
        for gain_threshold in gain_thresholds:
            labels = (target > gain_threshold).astype(np.int8)
            if len(set(labels[train_candidates].tolist())) < 2:
                continue
            for alpha in (1e-5, 1e-4, 1e-3):
                model = SGDClassifier(
                    loss="log_loss", penalty="l2", alpha=alpha,
                    class_weight="balanced", max_iter=300, tol=1e-5,
                    random_state=42, average=True,
                )
                model.fit(x[train_candidates], labels[train_candidates])
                predictions = model.decision_function(x)
                fitted.append(({
                    "estimator": "classifier", "target": target_name,
                    "alpha": alpha, "gain_threshold": gain_threshold,
                }, model, predictions))
    return fitted


def serializable_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "selected_candidate_ids"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--semantic-model", type=Path)
    parser.add_argument(
        "--semantic-top-k", type=int,
        help="restrict utility learning and selection to semantic top-k candidates",
    )
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.01)
    parser.add_argument("--minimum-precision", type=float, default=0.6)
    parser.add_argument("--minimum-emitted", type=int, default=5)
    parser.add_argument(
        "--selection-objective",
        choices=("controlled_gain", "balanced_accuracy"),
        default="controlled_gain",
    )
    args = parser.parse_args()

    training_rows = load_jsonl(args.input)
    validation_rows = load_jsonl(args.validation_input)
    rows = training_rows + validation_rows
    semantic_model = load_semantic_model(args.semantic_model)
    if args.semantic_top_k is not None:
        if semantic_model is None:
            raise SystemExit("--semantic-top-k requires --semantic-model")
        if args.semantic_top_k < 1:
            raise SystemExit("--semantic-top-k must be positive")
        for row in rows:
            scores = semantic_scores(row, semantic_model)
            keep = np.argsort(-np.asarray(scores))[: args.semantic_top_k]
            row["candidates"] = [row["candidates"][int(index)] for index in keep]
    groups = [row["question_group"] for row in rows]
    training_groups = set(groups[: len(training_rows)])
    buckets = [group_bucket(group) for group in groups]
    train_rows = [index for index in range(len(training_rows)) if buckets[index] not in {0, 1}]
    calibration_rows = [index for index in range(len(training_rows)) if buckets[index] == 1]
    test_rows = [index for index in range(len(training_rows)) if buckets[index] == 0]
    frozen_rows = list(range(len(training_rows), len(rows)))
    novel_frozen_rows = [index for index in frozen_rows if groups[index] not in training_groups]
    seen_frozen_rows = [index for index in frozen_rows if groups[index] in training_groups]

    sweep = []
    trained: list[tuple[dict[str, Any], Any, np.ndarray, list[tuple[int, int]]]] = []
    for feature_mode in ("generic", "reader", "reader_semantic"):
        if feature_mode == "reader_semantic" and semantic_model is None:
            continue
        x, targets, row_slices = build_examples(
            rows, feature_mode, semantic_model
        )
        train_candidates = candidate_indices(row_slices, train_rows)
        calibration_candidates = candidate_indices(row_slices, calibration_rows)
        for specification, model, predictions in fit_models(
            x, targets, train_candidates
        ):
            threshold, calibration = choose_threshold(
                rows, predictions, row_slices, calibration_rows,
                minimum_gain=args.minimum_gain,
                minimum_precision=args.minimum_precision,
                minimum_emitted=args.minimum_emitted,
                selection_objective=args.selection_objective,
            )
            target = targets[specification["target"]]
            positive = (target > args.minimum_gain).astype(np.int8)
            entry = specification | {
                "feature_mode": feature_mode,
                "threshold": threshold,
                "calibration_candidate_auroc": safe_metric(
                    roc_auc_score, positive[calibration_candidates],
                    predictions[calibration_candidates],
                ),
                "calibration_candidate_ap": safe_metric(
                    average_precision_score, positive[calibration_candidates],
                    predictions[calibration_candidates],
                ),
                "calibration": serializable_report(calibration),
                "test": serializable_report(selection_report(
                    rows, predictions, row_slices, test_rows, threshold,
                    minimum_gain=args.minimum_gain,
                )),
            }
            sweep.append(entry)
            trained.append((entry, model, predictions, row_slices))

    primary_key = (
        "balanced_controlled_gain"
        if args.selection_objective == "controlled_gain"
        else "balanced_accuracy_delta"
    )
    selected_index = max(range(len(sweep)), key=lambda index: (
        sweep[index]["calibration"][primary_key] or 0.0,
        sweep[index]["calibration"]["balanced_controlled_gain"],
        sweep[index]["calibration"]["balanced_raw_gain"],
        sweep[index]["calibration_candidate_ap"],
    ))
    selected, model, predictions, row_slices = trained[selected_index]
    threshold = float(selected["threshold"])
    _, targets, _ = build_examples(
        rows, selected["feature_mode"], semantic_model
    )
    raw_target = targets["utility"]
    controlled_target = targets["controlled_utility"]

    def evaluate_rows(ids: list[int]) -> dict[str, Any]:
        return serializable_report(selection_report(
            rows, predictions, row_slices, ids, threshold,
            minimum_gain=args.minimum_gain,
        ))

    frozen = evaluate_rows(frozen_rows)
    frozen["novel_question_groups"] = evaluate_rows(novel_frozen_rows)
    frozen["seen_question_groups"] = evaluate_rows(seen_frozen_rows)
    oracle_raw = oracle_predictions(rows, row_slices, raw_target)
    oracle_controlled = oracle_predictions(rows, row_slices, controlled_target)
    frozen["oracle_raw"] = serializable_report(selection_report(
        rows, oracle_raw, row_slices, frozen_rows, 0.0,
        minimum_gain=args.minimum_gain,
    ))
    frozen["oracle_controlled"] = serializable_report(selection_report(
        rows, oracle_controlled, row_slices, frozen_rows, 0.0,
        minimum_gain=args.minimum_gain,
    ))

    coefficients = np.asarray(model.coef_, dtype=np.float32).reshape(1, -1)
    intercept = np.asarray(model.intercept_, dtype=np.float32).reshape(-1)
    if selected["estimator"] == "ridge":
        coefficients *= float(selected["target_scale"])
        intercept *= float(selected["target_scale"])
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "weights": coefficients.astype(np.float16),
        "intercept": intercept,
        "threshold": np.asarray([threshold], dtype=np.float32),
        "feature_mode": np.asarray([selected["feature_mode"]]),
        "estimator": np.asarray([selected["estimator"]]),
        "target": np.asarray([selected["target"]]),
        "semantic_top_k": np.asarray(
            [-1 if args.semantic_top_k is None else args.semantic_top_k],
            dtype=np.int32,
        ),
    }
    if semantic_model is not None:
        payload["semantic_weights"] = semantic_model["weights"].astype(np.float16)
        payload["semantic_intercept"] = np.asarray(
            [semantic_model["intercept"]], dtype=np.float32
        )
    np.savez_compressed(args.output_model, **payload)
    report = {
        "training_rows": len(training_rows),
        "validation_rows": len(validation_rows),
        "split_rows": {
            "train": len(train_rows), "calibration": len(calibration_rows),
            "test": len(test_rows), "frozen": len(frozen_rows),
            "frozen_novel": len(novel_frozen_rows),
        },
        "candidate_counts": Counter(
            len(row["candidates"]) for row in rows
        ),
        "minimum_gain": args.minimum_gain,
        "minimum_precision": args.minimum_precision,
        "semantic_top_k": args.semantic_top_k,
        "selection_objective": args.selection_objective,
        "sweep": sweep,
        "selected": selected,
        "frozen_validation": frozen,
        "model_bytes": args.output_model.stat().st_size,
    }
    args.output_report.write_text(
        json.dumps(report, indent=2, default=lambda value: dict(value)) + "\n"
    )
    print(json.dumps({
        "selected": selected,
        "frozen_validation": frozen,
        "model_bytes": report["model_bytes"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
