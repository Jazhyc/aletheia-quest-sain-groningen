#!/usr/bin/env python3
"""Fit a leakage-safe sparse n-gram stack over heterogeneous adapter votes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pid_specialist_ensemble.analyze_ensemble import (
    balanced_cell_weights,
    comparison_counts,
    load_member_frame,
    load_selection_keys,
    metrics,
    parse_member,
)
from experiments.text_probe.run_text_probe import load_split


def make_vectorizer() -> TfidfVectorizer:
    """Return the single frozen sparse text representation for this ablation."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=20_000,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )


def make_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=2_000,
        solver="liblinear",
        random_state=0,
    )


def append_member_features(text_features: Any, members: np.ndarray) -> Any:
    """Append three binary member columns to a sparse text feature matrix."""
    return hstack([text_features, csr_matrix(members)], format="csr")


def attach_text(frame: pd.DataFrame, text_frame: pd.DataFrame) -> pd.DataFrame:
    """Join inference-visible transcript text with exact label validation."""
    text = text_frame[["dataset", "index", "label", "output_context"]].copy()
    text["index"] = text["index"].astype(str)
    text = text.rename(columns={"label": "text_label", "output_context": "text"})
    merged = frame.merge(text, on=["dataset", "index"], validate="one_to_one")
    if len(merged) != len(frame):
        raise ValueError("text rows do not cover every member prediction")
    if not merged["label"].eq(merged.pop("text_label")).all():
        raise ValueError("text and member artifacts disagree on labels")
    return merged


def fit_scores(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    member_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit adapter-only, text-only, and hybrid models on one training fold."""
    weights = balanced_cell_weights(train)
    train_members = train[member_names].to_numpy(dtype=float)
    evaluation_members = evaluation[member_names].to_numpy(dtype=float)
    labels = train["label"].to_numpy(dtype=int)

    adapter_model = make_classifier()
    adapter_model.fit(train_members, labels, sample_weight=weights)

    vectorizer = make_vectorizer()
    train_text = vectorizer.fit_transform(train["text"])
    evaluation_text = vectorizer.transform(evaluation["text"])

    text_model = make_classifier()
    text_model.fit(train_text, labels, sample_weight=weights)

    hybrid_model = make_classifier()
    hybrid_model.fit(
        append_member_features(train_text, train_members),
        labels,
        sample_weight=weights,
    )
    scores = {
        "adapter_only": adapter_model.predict_proba(evaluation_members)[:, 1],
        "ngram_only": text_model.predict_proba(evaluation_text)[:, 1],
        "hybrid": hybrid_model.predict_proba(
            append_member_features(evaluation_text, evaluation_members)
        )[:, 1],
    }
    feature_names = vectorizer.get_feature_names_out()
    text_coefficients = hybrid_model.coef_[0][:-len(member_names)]
    top_count = min(20, len(feature_names))
    positive = np.argsort(text_coefficients)[-top_count:][::-1]
    negative = np.argsort(text_coefficients)[:top_count]
    metadata = {
        "vocabulary_size": len(feature_names),
        "member_coefficients": dict(zip(
            member_names,
            hybrid_model.coef_[0][-len(member_names):].tolist(),
            strict=True,
        )),
        "intercept": float(hybrid_model.intercept_[0]),
        "top_positive_ngrams": [
            [str(feature_names[index]), float(text_coefficients[index])]
            for index in positive
        ],
        "top_negative_ngrams": [
            [str(feature_names[index]), float(text_coefficients[index])]
            for index in negative
        ],
    }
    return scores, metadata


def grouped_oof_scores(
    frame: pd.DataFrame,
    member_names: list[str],
) -> dict[str, np.ndarray]:
    """Score every training row while holding its entire dataset unit out."""
    scores = {
        name: np.full(len(frame), np.nan, dtype=float)
        for name in ("adapter_only", "ngram_only", "hybrid")
    }
    for dataset in sorted(frame["dataset"].unique()):
        held_out = frame["dataset"].eq(dataset).to_numpy()
        fold_scores, _ = fit_scores(
            frame.loc[~held_out].reset_index(drop=True),
            frame.loc[held_out].reset_index(drop=True),
            member_names,
        )
        for name, values in fold_scores.items():
            scores[name][held_out] = values
    if any(np.isnan(values).any() for values in scores.values()):
        raise RuntimeError("grouped OOF scoring left unfilled rows")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-member", action="append", required=True)
    parser.add_argument("--validation-member", action="append", required=True)
    parser.add_argument("--selection-manifest", action="append", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--max-context-chars", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_specs = [parse_member(value) for value in args.train_member]
    validation_specs = [parse_member(value) for value in args.validation_member]
    member_names = [name for name, _ in train_specs]
    if [name for name, _ in validation_specs] != member_names:
        raise ValueError("train and validation member names/order must match")

    train = load_member_frame(train_specs).reset_index(drop=True)
    validation = load_member_frame(validation_specs).reset_index(drop=True)
    selected = load_selection_keys([path.resolve() for path in args.selection_manifest])
    leaked = pd.Series([
        (str(dataset), str(index)) in selected
        for dataset, index in zip(train["dataset"], train["index"], strict=True)
    ])
    train = train.loc[~leaked].reset_index(drop=True)

    train_text = load_split(
        "train", args.splits_dir.resolve(), max_context_chars=args.max_context_chars
    ).frame
    validation_text = load_split(
        "validation", args.splits_dir.resolve(), max_context_chars=args.max_context_chars
    ).frame
    train = attach_text(train, train_text)
    validation = attach_text(validation, validation_text)

    oof_scores = grouped_oof_scores(train, member_names)
    validation_scores, model_metadata = fit_scores(train, validation, member_names)
    labels = validation["label"].to_numpy(dtype=int)
    deception_scores = validation["deception"].to_numpy(dtype=float)
    report = {
        "protocol": {
            "text_view": "output_context_without_reasoning",
            "max_context_chars": args.max_context_chars,
            "vectorizer": {
                "analyzer": "word",
                "ngram_range": [1, 2],
                "min_df": 3,
                "max_df": 0.98,
                "max_features": 20_000,
                "sublinear_tf": True,
            },
            "classifier": {"type": "logistic_regression", "c": 1.0, "threshold": 0.5},
            "grouped_oof": "leave_one_dataset_unit_out",
            "member_order": member_names,
            "base_training_rows_excluded": int(leaked.sum()),
            "meta_training_rows": len(train),
        },
        "grouped_oof": {
            name: metrics(train, values) for name, values in oof_scores.items()
        },
        "validation": {
            name: metrics(validation, values)
            for name, values in validation_scores.items()
        },
        "validation_comparison": {
            name: {
                "versus_deception": comparison_counts(labels, deception_scores, values),
                "versus_adapter_logistic": comparison_counts(
                    labels, validation_scores["adapter_only"], values
                ),
            }
            for name, values in validation_scores.items()
        },
        "model": model_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
