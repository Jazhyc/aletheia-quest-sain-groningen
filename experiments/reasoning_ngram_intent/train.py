#!/usr/bin/env python3
"""Train a leakage-resistant reasoning-only n-gram intent classifier."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path
import re
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pid_specialist_ensemble.analyze_ensemble import (  # noqa: E402
    metrics_without_groups,
)
from experiments.privileged_information_distillation.core import (  # noqa: E402
    final_assistant_reasoning,
    truncate_text,
)
from experiments.text_probe.run_text_probe import (  # noqa: E402
    load_labels,
    load_split_config,
)


HF_SPLIT = "test"
MAX_FEATURES = 20_000
THRESHOLD = 0.5
INTENT_CUE_RE = re.compile(
    r"(?i)\b(?:deceiv\w*|dishonest\w*|fals(?:e|ely|ify\w*)|"
    r"lie|lies|lied|lying|mislead\w*|trick\w*|manipulat\w*|"
    r"wrong\s+answer|incorrect\s+answer)\b"
)


@dataclasses.dataclass(frozen=True)
class Candidate:
    truncation: str
    max_chars: int
    min_df: int
    c: float


def candidate_grid() -> list[Candidate]:
    """Return the preregistered capacity/regularization grid."""
    return [
        Candidate(truncation=truncation, max_chars=4_000, min_df=min_df, c=c)
        for truncation in ("tail", "head_tail")
        for min_df in (3, 8)
        for c in (0.03, 0.1, 0.3, 1.0)
    ]


def make_vectorizer(candidate: Candidate) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=candidate.min_df,
        max_df=0.98,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )


def make_classifier(c: float) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        max_iter=2_000,
        solver="liblinear",
        random_state=0,
    )


def balanced_dataset_label_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give each dataset-unit/label cell equal total training weight."""
    cells = pd.Series(
        list(zip(frame["dataset"], frame["label"].astype(int), strict=True))
    )
    counts = cells.value_counts()
    return np.asarray([
        len(frame) / (len(counts) * counts[cell])
        for cell in cells
    ])


def reasoning_view(reasoning: str, candidate: Candidate) -> str:
    return truncate_text(reasoning, candidate.max_chars, candidate.truncation)


def load_reasoning_split(split: str, splits_dir: Path) -> pd.DataFrame:
    """Load only varied-deception rows and their final assistant reasoning."""
    configs = [
        cfg
        for cfg in load_split_config(
            splits_dir / f"dry.{split}.yaml",
            ROOT,
        )
        if "varied-deception" in cfg.name
    ]
    if not configs:
        raise ValueError(f"{split}: no varied-deception datasets")

    from datasets import load_dataset

    frames: list[pd.DataFrame] = []
    for cfg in configs:
        labels = load_labels(cfg)
        label_by_index = dict(
            zip(labels["index"], labels["label"].astype(int), strict=True)
        )
        wanted = set(label_by_index)
        dataset = load_dataset(cfg.name, split=HF_SPLIT)
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))

        rows = []
        for row in dataset:
            index = row["index"]
            if index not in wanted:
                continue
            reasoning = final_assistant_reasoning(row["messages"]).strip()
            rows.append({
                "dataset": cfg.name,
                "index": str(index),
                "label": int(label_by_index[index]),
                "reasoning": reasoning,
                "reasoning_chars": len(reasoning),
                "reasoning_words": len(reasoning.split()),
            })
        if len(rows) != len(labels):
            raise RuntimeError(
                f"{cfg.name}: loaded {len(rows)} rows for {len(labels)} labels"
            )
        frames.append(pd.DataFrame(rows))

    return pd.concat(frames, ignore_index=True)


def fit_candidate(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    candidate: Candidate,
) -> tuple[np.ndarray, TfidfVectorizer, LogisticRegression]:
    vectorizer, classifier = fit_model(train, candidate)
    evaluation_text = [
        reasoning_view(value, candidate)
        for value in evaluation["reasoning"]
    ]
    evaluation_features = vectorizer.transform(evaluation_text)
    return (
        classifier.predict_proba(evaluation_features)[:, 1],
        vectorizer,
        classifier,
    )


def fit_model(
    train: pd.DataFrame,
    candidate: Candidate,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Fit one frozen representation and classifier."""
    train_text = [
        reasoning_view(value, candidate)
        for value in train["reasoning"]
    ]
    vectorizer = make_vectorizer(candidate)
    train_features = vectorizer.fit_transform(train_text)
    classifier = make_classifier(candidate.c)
    classifier.fit(
        train_features,
        train["label"].to_numpy(dtype=int),
        sample_weight=balanced_dataset_label_weights(train),
    )
    return vectorizer, classifier


def grouped_oof_scores(
    frame: pd.DataFrame,
    candidate: Candidate,
) -> np.ndarray:
    """Hold out each complete varied dataset unit in turn."""
    scores = np.full(len(frame), np.nan, dtype=float)
    for dataset in sorted(frame["dataset"].unique()):
        held_out = frame["dataset"].eq(dataset).to_numpy()
        fold_scores, _, _ = fit_candidate(
            frame.loc[~held_out].reset_index(drop=True),
            frame.loc[held_out].reset_index(drop=True),
            candidate,
        )
        scores[held_out] = fold_scores
    if np.isnan(scores).any():
        raise RuntimeError("grouped OOF scoring left unfilled rows")
    return scores


def length_only_oof_scores(frame: pd.DataFrame) -> np.ndarray:
    """A low-capacity control using only reasoning length."""
    features = np.column_stack([
        np.log1p(frame["reasoning_chars"].to_numpy(dtype=float)),
        np.log1p(frame["reasoning_words"].to_numpy(dtype=float)),
    ])
    scores = np.full(len(frame), np.nan, dtype=float)
    for dataset in sorted(frame["dataset"].unique()):
        held_out = frame["dataset"].eq(dataset).to_numpy()
        classifier = make_classifier(1.0)
        classifier.fit(
            features[~held_out],
            frame.loc[~held_out, "label"].to_numpy(dtype=int),
            sample_weight=balanced_dataset_label_weights(
                frame.loc[~held_out].reset_index(drop=True)
            ),
        )
        scores[held_out] = classifier.predict_proba(features[held_out])[:, 1]
    return scores


def score_model(
    frame: pd.DataFrame,
    candidate: Candidate,
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    *,
    redact_intent_cues: bool = False,
) -> np.ndarray:
    text = [
        reasoning_view(value, candidate)
        for value in frame["reasoning"]
    ]
    if redact_intent_cues:
        text = [INTENT_CUE_RE.sub("[INTENT_CUE]", value) for value in text]
    return classifier.predict_proba(vectorizer.transform(text))[:, 1]


def top_ngrams(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    count: int = 30,
) -> dict[str, list[list[Any]]]:
    names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    count = min(count, len(names))
    positive = np.argsort(coefficients)[-count:][::-1]
    negative = np.argsort(coefficients)[:count]
    return {
        "positive": [
            [str(names[index]), float(coefficients[index])]
            for index in positive
        ],
        "negative": [
            [str(names[index]), float(coefficients[index])]
            for index in negative
        ],
    }


def metric_summary(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    return metrics_without_groups(frame.reset_index(drop=True), scores)


def candidate_key(
    candidate: Candidate,
    result: dict[str, Any],
) -> tuple[float, float, float, float, int, int]:
    """Prefer OOF accuracy, then ranking/precision, then simpler models."""
    return (
        result["balanced_accuracy"],
        result["auroc"],
        -result["fpr"],
        -candidate.c,
        candidate.min_df,
        int(candidate.truncation == "tail"),
    )


def write_predictions(
    path: Path,
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "dataset": frame["dataset"],
        "index": frame["index"],
        "label": frame["label"],
        "score": scores,
        "deceptive": scores >= THRESHOLD,
    }).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=ROOT / "dev_splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "blackbox"
            / "reasoning_ngram_intent_v1"
        ),
    )
    args = parser.parse_args()

    train = load_reasoning_split("train", args.splits_dir.resolve())
    validation = load_reasoning_split("validation", args.splits_dir.resolve())
    print(
        f"train rows={len(train)} datasets={train['dataset'].nunique()} "
        f"empty_reasoning={int(train['reasoning'].eq('').sum())}"
    )
    print(
        f"validation rows={len(validation)} datasets={validation['dataset'].nunique()} "
        f"empty_reasoning={int(validation['reasoning'].eq('').sum())}"
    )
    if train["reasoning"].eq("").any() or validation["reasoning"].eq("").any():
        raise ValueError("varied-deception reasoning unexpectedly contains empty rows")

    grid = []
    best: tuple[Candidate, np.ndarray, dict[str, Any]] | None = None
    for number, candidate in enumerate(candidate_grid(), start=1):
        print(f"[{number}/{len(candidate_grid())}] {candidate}", flush=True)
        scores = grouped_oof_scores(train, candidate)
        result = metric_summary(train, scores)
        grid.append({**dataclasses.asdict(candidate), **result})
        if (
            best is None
            or candidate_key(candidate, result)
            > candidate_key(best[0], best[2])
        ):
            best = candidate, scores, result
    if best is None:
        raise RuntimeError("no candidates evaluated")

    selected, oof_scores, oof_metrics = best
    validation_scores, vectorizer, classifier = fit_candidate(
        train,
        validation,
        selected,
    )
    cue_redacted_scores = score_model(
        validation,
        selected,
        vectorizer,
        classifier,
        redact_intent_cues=True,
    )
    length_scores = length_only_oof_scores(train)
    validation_metrics = metric_summary(validation, validation_scores)
    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "input": "final_assistant_reasoning_only",
            "scenario": "varied-deception only",
            "selection": "leave-one-dataset-unit-out grouped OOF on train only",
            "validation_used_for_selection": False,
            "threshold": THRESHOLD,
            "sample_weighting": "equal total weight per dataset-unit/label cell",
            "max_features": MAX_FEATURES,
            "ngram_range": [1, 2],
        },
        "data": {
            "train_rows": len(train),
            "train_datasets": int(train["dataset"].nunique()),
            "validation_rows": len(validation),
            "validation_datasets": int(validation["dataset"].nunique()),
            "train_empty_reasoning": int(train["reasoning"].eq("").sum()),
            "validation_empty_reasoning": int(
                validation["reasoning"].eq("").sum()
            ),
        },
        "selected_candidate": dataclasses.asdict(selected),
        "grouped_oof": oof_metrics,
        "grouped_oof_length_only_control": metric_summary(train, length_scores),
        "validation": validation_metrics,
        "validation_intent_cues_redacted": metric_summary(
            validation,
            cue_redacted_scores,
        ),
        "model": {
            "vocabulary_size": len(vectorizer.vocabulary_),
            "intercept": float(classifier.intercept_[0]),
            "top_ngrams": top_ngrams(vectorizer, classifier),
        },
        "grid": sorted(
            grid,
            key=lambda row: (
                row["balanced_accuracy"],
                row["auroc"],
                -row["fpr"],
            ),
            reverse=True,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    joblib.dump(
        {
            "candidate": dataclasses.asdict(selected),
            "threshold": THRESHOLD,
            "vectorizer": vectorizer,
            "classifier": classifier,
        },
        args.output_dir / "model.joblib",
    )
    write_predictions(
        args.output_dir / "predictions" / "train_grouped_oof.csv",
        train,
        oof_scores,
    )
    write_predictions(
        args.output_dir / "predictions" / "validation.csv",
        validation,
        validation_scores,
    )
    pd.DataFrame(grid).to_csv(args.output_dir / "grid.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
