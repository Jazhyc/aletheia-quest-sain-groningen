#!/usr/bin/env python3
"""Evaluate fixed-OR and grouped logistic Phoenix/reasoning ensembles."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pid_specialist_ensemble.analyze_ensemble import (  # noqa: E402
    comparison_counts,
    metrics_without_groups,
)
from experiments.reasoning_ngram_intent.train import (  # noqa: E402
    balanced_dataset_label_weights,
)


KEYS = ["dataset", "index"]
FEATURES = ["phoenix_score", "reasoning_ngram_score"]
THRESHOLD = 0.5
CANDIDATE_CS = (0.01, 0.03, 0.1, 0.3, 1.0)


@dataclasses.dataclass(frozen=True)
class Selection:
    c: float
    scores: np.ndarray
    metrics: dict[str, Any]


def load_phoenix(path: Path) -> pd.DataFrame:
    """Load cached Phoenix generations and retain varied rows only."""
    frame = pd.read_json(path, lines=True)
    required = {*KEYS, "label", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing Phoenix columns {sorted(missing)}")
    if "parse_error" not in frame:
        frame["parse_error"] = False
    frame = frame.loc[
        frame["dataset"].str.contains("varied-deception", regex=False),
        [*KEYS, "label", "score", "parse_error"],
    ].copy()
    frame["index"] = frame["index"].astype(str)
    frame = frame.rename(columns={
        "score": "phoenix_score",
        "parse_error": "phoenix_parse_error",
    })
    return frame


def load_reasoning_ngram(path: Path) -> pd.DataFrame:
    """Load reasoning-only n-gram predictions."""
    frame = pd.read_csv(path)
    required = {*KEYS, "label", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing n-gram columns {sorted(missing)}")
    frame = frame[[*KEYS, "label", "score"]].copy()
    frame["index"] = frame["index"].astype(str)
    return frame.rename(columns={"score": "reasoning_ngram_score"})


def load_joined(phoenix_path: Path, ngram_path: Path) -> pd.DataFrame:
    """Join members exactly and reject incomplete or label-mismatched inputs."""
    phoenix = load_phoenix(phoenix_path)
    ngram = load_reasoning_ngram(ngram_path)
    frame = phoenix.merge(
        ngram,
        on=KEYS,
        how="outer",
        suffixes=("_phoenix", "_ngram"),
        indicator=True,
        validate="one_to_one",
    )
    incomplete = frame["_merge"].ne("both")
    if incomplete.any():
        examples = frame.loc[incomplete, [*KEYS, "_merge"]].head().to_dict("records")
        raise ValueError(f"member key mismatch: {examples}")
    if not frame["label_phoenix"].eq(frame["label_ngram"]).all():
        raise ValueError("Phoenix and n-gram labels disagree")
    frame["label"] = frame.pop("label_phoenix").astype(int)
    frame = frame.drop(columns=["label_ngram", "_merge"])
    return frame.sort_values(KEYS).reset_index(drop=True)


def make_model(c: float) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        max_iter=2_000,
        solver="liblinear",
        random_state=0,
    )


def fit_model(frame: pd.DataFrame, c: float) -> LogisticRegression:
    model = make_model(c)
    model.fit(
        frame[FEATURES].to_numpy(dtype=float),
        frame["label"].to_numpy(dtype=int),
        sample_weight=balanced_dataset_label_weights(frame),
    )
    return model


def grouped_oof_scores(frame: pd.DataFrame, c: float) -> np.ndarray:
    """Fit on all but one complete dataset unit and score the held-out unit."""
    scores = np.full(len(frame), np.nan, dtype=float)
    for dataset in sorted(frame["dataset"].unique()):
        held_out = frame["dataset"].eq(dataset).to_numpy()
        model = fit_model(frame.loc[~held_out].reset_index(drop=True), c)
        scores[held_out] = model.predict_proba(
            frame.loc[held_out, FEATURES].to_numpy(dtype=float)
        )[:, 1]
    if np.isnan(scores).any():
        raise RuntimeError("grouped OOF scoring left unfilled rows")
    return scores


def selection_key(selection: Selection) -> tuple[float, float, float, float]:
    return (
        selection.metrics["balanced_accuracy"],
        selection.metrics["auroc"],
        -selection.metrics["fpr"],
        -selection.c,
    )


def select_c(frame: pd.DataFrame) -> Selection:
    """Select regularization using grouped OOF on the provided training rows."""
    selections = []
    for c in CANDIDATE_CS:
        scores = grouped_oof_scores(frame, c)
        selections.append(Selection(
            c=c,
            scores=scores,
            metrics=metrics_without_groups(frame, scores),
        ))
    return max(selections, key=selection_key)


def nested_grouped_oof_scores(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate the full select-then-fit procedure without scoring fit rows."""
    scores = np.full(len(frame), np.nan, dtype=float)
    selected_by_outer_fold: dict[str, float] = {}
    for dataset in sorted(frame["dataset"].unique()):
        held_out = frame["dataset"].eq(dataset).to_numpy()
        outer_train = frame.loc[~held_out].reset_index(drop=True)
        selected = select_c(outer_train)
        model = fit_model(outer_train, selected.c)
        scores[held_out] = model.predict_proba(
            frame.loc[held_out, FEATURES].to_numpy(dtype=float)
        )[:, 1]
        selected_by_outer_fold[dataset] = selected.c
    if np.isnan(scores).any():
        raise RuntimeError("nested grouped OOF scoring left unfilled rows")
    return scores, selected_by_outer_fold


def or_scores(frame: pd.DataFrame) -> np.ndarray:
    """Continuous max score with binary behavior equal to logical OR."""
    return frame[FEATURES].max(axis=1).to_numpy(dtype=float)


def per_dataset(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, dict[str, Any]]:
    scored = frame[[*KEYS, "label"]].copy()
    scored["score"] = scores
    return {
        dataset: metrics_without_groups(group.reset_index(drop=True), group["score"].to_numpy())
        for dataset, group in scored.groupby("dataset", sort=True)
    }


def score_bundle(
    frame: pd.DataFrame,
    logistic_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "phoenix": frame["phoenix_score"].to_numpy(dtype=float),
        "reasoning_ngram": frame["reasoning_ngram_score"].to_numpy(dtype=float),
        "or": or_scores(frame),
        "logistic": logistic_scores,
    }


def summarize(
    frame: pd.DataFrame,
    scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=int)
    return {
        "metrics": {
            name: metrics_without_groups(frame, values)
            for name, values in scores.items()
        },
        "comparisons_to_phoenix": {
            name: comparison_counts(labels, scores["phoenix"], values)
            for name, values in scores.items()
            if name != "phoenix"
        },
        "per_dataset": {
            name: per_dataset(frame, values)
            for name, values in scores.items()
        },
        "phoenix_parse_errors": int(frame["phoenix_parse_error"].sum()),
        "member_decision_disagreements": int(
            (
                (scores["phoenix"] >= THRESHOLD)
                != (scores["reasoning_ngram"] >= THRESHOLD)
            ).sum()
        ),
    }


def write_predictions(
    path: Path,
    frame: pd.DataFrame,
    scores: dict[str, np.ndarray],
) -> None:
    output = frame[[*KEYS, "label", *FEATURES, "phoenix_parse_error"]].copy()
    for name, values in scores.items():
        output[f"{name}_ensemble_score"] = values
        output[f"{name}_ensemble_deceptive"] = values >= THRESHOLD
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-phoenix", type=Path, required=True)
    parser.add_argument("--validation-ngram", type=Path, required=True)
    parser.add_argument("--test-phoenix", type=Path)
    parser.add_argument("--test-ngram", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.test_phoenix is None) != (args.test_ngram is None):
        parser.error("provide both test members or neither")

    validation = load_joined(args.validation_phoenix, args.validation_ngram)
    nested_scores, outer_selections = nested_grouped_oof_scores(validation)
    final_selection = select_c(validation)
    final_model = fit_model(validation, final_selection.c)
    validation_scores = score_bundle(validation, nested_scores)

    report: dict[str, Any] = {
        "protocol": {
            "members": FEATURES,
            "meta_training_split": "validation varied-deception only",
            "meta_evaluation": "nested leave-one-dataset-unit-out OOF",
            "threshold": THRESHOLD,
            "or_rule": "max member score; binary decision is logical OR",
            "logistic_candidates_c": list(CANDIDATE_CS),
            "sample_weighting": "equal total weight per dataset-unit/label cell",
            "test_used_for_selection": False,
        },
        "validation_nested_oof": summarize(validation, validation_scores),
        "meta_model": {
            "feature_order": FEATURES,
            "selected_c": final_selection.c,
            "selection_grouped_oof_metrics": final_selection.metrics,
            "selected_c_by_outer_fold": outer_selections,
            "coefficient": final_model.coef_[0].tolist(),
            "intercept": float(final_model.intercept_[0]),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(
        args.output_dir / "validation_nested_oof.csv",
        validation,
        validation_scores,
    )
    artifact = {
        "feature_order": FEATURES,
        "threshold": THRESHOLD,
        "selected_c": final_selection.c,
        "classifier": final_model,
        "fit": "all varied validation rows after nested grouped OOF evaluation",
    }
    joblib.dump(artifact, args.output_dir / "meta_model.joblib")

    if args.test_phoenix is not None:
        test = load_joined(args.test_phoenix, args.test_ngram)
        logistic_scores = final_model.predict_proba(
            test[FEATURES].to_numpy(dtype=float)
        )[:, 1]
        test_scores = score_bundle(test, logistic_scores)
        report["frozen_test"] = summarize(test, test_scores)
        write_predictions(args.output_dir / "test.csv", test, test_scores)

    (args.output_dir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
