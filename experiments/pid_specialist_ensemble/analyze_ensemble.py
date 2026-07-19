#!/usr/bin/env python3
"""Fit and evaluate fixed binary specialist ensemble rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


KEYS = ["dataset", "index"]


def parse_member(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    return name, Path(path).resolve()


def load_member_frame(specs: list[tuple[str, Path]]) -> pd.DataFrame:
    """Join binary member predictions with exact key/label validation."""
    merged = None
    for name, path in specs:
        frame = pd.read_json(path, lines=True)
        if "parse_error" not in frame:
            frame["parse_error"] = False
        frame = frame[[*KEYS, "label", "score", "parse_error"]]
        frame["index"] = frame["index"].astype(str)
        frame = frame.rename(columns={
            "score": name,
            "label": f"label_{name}",
            "parse_error": f"parse_error_{name}",
        })
        merged = frame if merged is None else merged.merge(
            frame, on=KEYS, validate="one_to_one"
        )
    if merged is None:
        raise ValueError("at least one member is required")
    label_columns = [column for column in merged if column.startswith("label_")]
    if not merged[label_columns].nunique(axis=1).eq(1).all():
        raise ValueError("member artifacts disagree on labels")
    merged["label"] = merged.pop(label_columns[0]).astype(int)
    merged = merged.drop(columns=label_columns[1:])
    for name, _ in specs:
        merged[name] = (merged[name].fillna(0.0) >= 0.5).astype(float)
    return merged


def scenario(dataset: str) -> str:
    return "varied" if "varied-deception" in dataset else "instructed"


def balanced_cell_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every scenario/label cell equal total meta-training weight."""
    cells = pd.Series([
        (scenario(dataset), int(label))
        for dataset, label in zip(frame["dataset"], frame["label"], strict=True)
    ])
    counts = cells.value_counts()
    return np.array([len(frame) / (len(counts) * counts[cell]) for cell in cells])


def metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    result = metrics_without_groups(frame, scores)
    result["scenarios"] = {
        name: metrics_without_groups(group, scores[group.index.to_numpy()])
        for name, group in frame.assign(
            _scenario=frame["dataset"].map(scenario)
        ).groupby("_scenario", sort=True)
    }
    return result


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = scores >= 0.5
    positives = labels == 1
    negatives = ~positives
    recall = float(predictions[positives].mean())
    fpr = float(predictions[negatives].mean())
    return {
        "balanced_accuracy": (recall + 1.0 - fpr) / 2.0,
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": recall,
        "fpr": fpr,
    }


def metrics_without_groups(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    """Return the project-standard dataset macro metrics plus pooled diagnostics."""
    scored = frame[["dataset", "label"]].copy()
    scored["_score"] = scores
    per_dataset = [
        binary_metrics(
            group["label"].to_numpy(dtype=int),
            group["_score"].to_numpy(dtype=float),
        )
        for _, group in scored.groupby("dataset", sort=True)
    ]
    result = {
        key: float(np.mean([value[key] for value in per_dataset]))
        for key in ("balanced_accuracy", "auroc", "recall", "fpr")
    }
    result["rows"] = len(frame)
    result["pooled"] = binary_metrics(
        scored["label"].to_numpy(dtype=int),
        scored["_score"].to_numpy(dtype=float),
    )
    return result


def comparison_counts(
    labels: np.ndarray,
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
) -> dict[str, int]:
    """Count candidate corrections and regressions against a reference."""
    reference_correct = (reference_scores >= 0.5) == labels
    candidate_correct = (candidate_scores >= 0.5) == labels
    return {
        "fixes": int((~reference_correct & candidate_correct).sum()),
        "breaks": int((reference_correct & ~candidate_correct).sum()),
        "decision_disagreements": int(
            ((reference_scores >= 0.5) != (candidate_scores >= 0.5)).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-member", action="append", required=True)
    parser.add_argument("--validation-member", action="append", required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_specs = [parse_member(value) for value in args.train_member]
    validation_specs = [parse_member(value) for value in args.validation_member]
    if [name for name, _ in train_specs] != [name for name, _ in validation_specs]:
        raise ValueError("train and validation member names/order must match")
    names = [name for name, _ in train_specs]
    train = load_member_frame(train_specs).reset_index(drop=True)
    validation = load_member_frame(validation_specs).reset_index(drop=True)

    selected = {
        (str(record["dataset"]), str(record["index"]))
        for record in (
            json.loads(line)
            for line in args.selection_manifest.read_text().splitlines()
            if line.strip()
        )
    }
    leakage_mask = pd.Series([
        (str(dataset), str(index)) in selected
        for dataset, index in zip(train["dataset"], train["index"], strict=True)
    ])
    meta_train = train.loc[~leakage_mask].reset_index(drop=True)
    features = meta_train[names].to_numpy(dtype=float)
    model = LogisticRegression(C=1.0, max_iter=1_000, random_state=0)
    model.fit(
        features,
        meta_train["label"].to_numpy(dtype=int),
        sample_weight=balanced_cell_weights(meta_train),
    )

    validation_features = validation[names].to_numpy(dtype=float)
    scores = {
        **{name: validation[name].to_numpy(dtype=float) for name in names},
        "or": validation_features.max(axis=1),
        "majority": (validation_features.sum(axis=1) >= 2).astype(float),
        "unanimity": (validation_features.sum(axis=1) == len(names)).astype(float),
        "logistic": model.predict_proba(validation_features)[:, 1],
    }
    train_probabilities = model.predict_proba(features)[:, 1]
    pairwise = {
        f"{left}__{right}": int((validation[left] != validation[right]).sum())
        for position, left in enumerate(names)
        for right in names[position + 1:]
    }
    report = {
        "members": names,
        "base_training_rows_excluded": int(leakage_mask.sum()),
        "meta_training_rows": len(meta_train),
        "meta_model": {
            "feature_order": names,
            "coefficient": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "regularization_c": 1.0,
            "threshold": 0.5,
            "sample_weighting": "equal scenario/label cells",
            "pattern_scores": {
                "".join(map(str, pattern)): float(model.predict_proba(
                    np.array(pattern, dtype=float).reshape(1, -1)
                )[0, 1])
                for pattern in np.ndindex(*(2,) * len(names))
            },
        },
        "pairwise_validation_disagreement": pairwise,
        "parse_errors": {
            "meta_train": {
                name: int(meta_train[f"parse_error_{name}"].sum())
                for name in names
            },
            "validation": {
                name: int(validation[f"parse_error_{name}"].sum())
                for name in names
            },
        },
        "meta_train": metrics(meta_train, train_probabilities),
        "validation": {
            name: metrics(validation, member_scores)
            for name, member_scores in scores.items()
        },
        "validation_comparisons_to_members": {
            ensemble: {
                member: comparison_counts(
                    validation["label"].to_numpy(dtype=int),
                    scores[member],
                    scores[ensemble],
                )
                for member in names
            }
            for ensemble in ("or", "majority", "unanimity", "logistic")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
