#!/usr/bin/env python3
"""Compare current local-v2.1 scores with the archived Phoenix 4.0 sweep."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_ROOT = (
    ROOT
    / "results/blackbox"
    / "qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1"
)
CURRENT_PREDICTIONS = (
    ROOT
    / "results/blackbox/phoenix_backend_parity_eunomia_v1/vllm/predictions.csv"
)
OUTPUT = (
    ROOT
    / "results/blackbox/phoenix_backend_parity_eunomia_v1"
    / "historical_local_v21_comparison.json"
)
DATASET = "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None"


def score_summary(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    scores = frame[column].to_numpy(dtype=np.float64)
    labels = frame["label"].to_numpy(dtype=int)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy_at_0_15": float(
            balanced_accuracy_score(labels, scores >= 0.15)
        ),
        "positive_rows_at_0_15": int(np.sum(scores >= 0.15)),
        "unique_scores": int(np.unique(scores).size),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_max": float(scores.max()),
    }


def pair_outcomes(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> np.ndarray:
    differences = positive_scores[:, None] - negative_scores[None, :]
    return np.sign(differences).astype(np.int8).ravel()


def pair_transition_report(
    frame: pd.DataFrame,
    old_column: str,
    new_column: str,
) -> dict[str, int]:
    positives = frame["label"].to_numpy(dtype=int) == 1
    old = pair_outcomes(
        frame.loc[positives, old_column].to_numpy(dtype=np.float64),
        frame.loc[~positives, old_column].to_numpy(dtype=np.float64),
    )
    new = pair_outcomes(
        frame.loc[positives, new_column].to_numpy(dtype=np.float64),
        frame.loc[~positives, new_column].to_numpy(dtype=np.float64),
    )
    names = {-1: "incorrect", 0: "tie", 1: "correct"}
    report = {"total_positive_negative_pairs": int(old.size)}
    for old_value, old_name in names.items():
        for new_value, new_name in names.items():
            report[f"{old_name}_to_{new_name}"] = int(
                np.sum((old == old_value) & (new == new_value))
            )
    return report


def compare_split(split: str, current: pd.DataFrame) -> dict:
    historical_path = (
        HISTORICAL_ROOT
        / f"{split}_phoenix_v3_auroc_margin_sweep_v1/generations.jsonl"
    )
    historical = pd.read_json(historical_path, lines=True)
    historical = historical.loc[
        historical["dataset"] == DATASET,
        ["dataset", "index", "label", "direct_margin_score"],
    ]
    current_columns = [
        "dataset",
        "index",
        "label",
        "score_base",
        "score_local_v21_adapter",
    ]
    merged = historical.merge(
        current[current_columns],
        on=["dataset", "index", "label"],
        validate="one_to_one",
    )
    if len(merged) != len(historical):
        raise RuntimeError(
            f"{split}: matched {len(merged)}/{len(historical)} historical rows"
        )

    old_column = "direct_margin_score"
    new_column = "score_local_v21_adapter"
    old_scores = merged[old_column].to_numpy(dtype=np.float64)
    new_scores = merged[new_column].to_numpy(dtype=np.float64)
    delta = new_scores - old_scores
    old_logits = np.log(old_scores / (1.0 - old_scores))
    new_logits = np.log(new_scores / (1.0 - new_scores))
    slope, intercept = np.polyfit(old_logits, new_logits, deg=1)
    fitted = slope * old_logits + intercept
    residual = new_logits - fitted
    base_scores = merged["score_base"].to_numpy(dtype=np.float64)
    base_delta = base_scores - old_scores

    return {
        "rows": int(len(merged)),
        "historical": score_summary(merged, old_column),
        "current_base": score_summary(merged, "score_base"),
        "current": score_summary(merged, new_column),
        "historical_vs_current_base": {
            "exact_equal_scores": int(np.sum(old_scores == base_scores)),
            "close_scores_at_1e_6": int(
                np.sum(np.isclose(old_scores, base_scores, atol=1e-6, rtol=0))
            ),
            "pearson": float(
                merged[[old_column, "score_base"]]
                .corr(method="pearson")
                .iloc[0, 1]
            ),
            "spearman": float(
                merged[[old_column, "score_base"]]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
            "mean_absolute_difference": float(np.abs(base_delta).mean()),
            "max_absolute_difference": float(np.abs(base_delta).max()),
        },
        "paired": {
            "exact_equal_scores": int(np.sum(old_scores == new_scores)),
            "close_scores_at_1e_6": int(
                np.sum(np.isclose(old_scores, new_scores, atol=1e-6, rtol=0))
            ),
            "pearson": float(
                merged[[old_column, new_column]]
                .corr(method="pearson")
                .iloc[0, 1]
            ),
            "spearman": float(
                merged[[old_column, new_column]]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
            "mean_signed_delta_current_minus_historical": float(delta.mean()),
            "mean_absolute_difference": float(np.abs(delta).mean()),
            "median_absolute_difference": float(np.median(np.abs(delta))),
            "max_absolute_difference": float(np.abs(delta).max()),
            "threshold_decision_flips_at_0_15": int(
                np.sum((old_scores >= 0.15) != (new_scores >= 0.15))
            ),
            "logit_affine_fit": {
                "slope": float(slope),
                "intercept": float(intercept),
                "residual_rmse": float(np.sqrt(np.mean(residual**2))),
            },
        },
        "positive_negative_pair_transitions": pair_transition_report(
            merged,
            old_column,
            new_column,
        ),
    }


def main() -> None:
    current = pd.read_csv(CURRENT_PREDICTIONS)
    if "score_local_v21_adapter" not in current:
        raise RuntimeError(
            "current predictions do not contain score_local_v21_adapter; "
            "run the updated vLLM parity job first"
        )
    report = {
        "dataset": DATASET,
        "historical_adapter_sha256": (
            "1407d88533513f348bcf5355b4dedbd94a263b085a71ae971889d1dd24d3978e"
        ),
        "splits": {
            split: compare_split(split, current)
            for split in ("validation", "test")
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
