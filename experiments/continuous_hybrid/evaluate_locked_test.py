#!/usr/bin/env python3
"""Fit the frozen continuous stack on validation and evaluate it once on local test."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.continuous_hybrid.stack_cached_features import (
    RESULTS,
    build_feature_frame,
    feature_sets,
    join_feature,
    read_jsonl,
    scenario_from_dataset,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import macro_metrics


DKS_TEST_METHOD = "continuous_hybrid_dks3072_logits_v1"
PHOENIX_TEST = RESULTS / (
    "qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/"
    "test_continuous_margin_v1/generations.jsonl"
)
GRPO_EMPTY_TEST = RESULTS / (
    "qwen_grpo_lora_r16_binary_t12_muonlr3e5_full_v2_logits_"
    "empty_reasoning_spaceprefix/test/predictions.csv"
)
GRPO_REASON_TEST = RESULTS / (
    "qwen_grpo_lora_r16_binary_t12_muonlr3e5_full_v2_logits_"
    "generated_reasoning/test/predictions.csv"
)


def load_rating_split(results: Path, method: str, prefix: str, split: str) -> pd.DataFrame:
    """Load and normalize one generated-rating run for an arbitrary split."""
    rows = pd.DataFrame(read_jsonl(results / method / split / "generations.jsonl"))
    rows["index"] = rows["index"].astype(str)
    rows["member"] = rows.get("ensemble_member", "single")
    rows["rating"] = rows["rating"].fillna(1).astype(float)
    pivot = rows.pivot(
        index=["dataset", "index"], columns="member", values="rating"
    ).reset_index()
    pivot.columns = [
        "dataset", "index", *(f"{prefix}_{column}" for column in pivot.columns[2:])
    ]
    columns = [column for column in pivot if column.startswith(f"{prefix}_")]
    pivot[columns] = (pivot[columns] - 1.0) / 6.0
    return pivot


def load_member_predictions(
    results: Path,
    method: str,
    split: str,
    member_features: dict[str, str],
) -> pd.DataFrame:
    """Load member scores persisted by a single ensemble judge run."""
    run_dir = results / method / split
    result = json.loads((run_dir / "result.json").read_text())
    merged: pd.DataFrame | None = None
    for member, feature in member_features.items():
        pieces = []
        for dataset_result in result["datasets"]:
            filename = f"{dataset_result['dataset'].replace('/', '__')}.csv"
            path = run_dir / "member_predictions" / member / filename
            part = pd.read_csv(path, usecols=["index", "score"])
            part["dataset"] = dataset_result["dataset"]
            pieces.append(part.rename(columns={"score": feature}))
        feature_frame = pd.concat(pieces, ignore_index=True)
        merged = feature_frame if merged is None else join_feature(merged, feature_frame)
    if merged is None:
        raise ValueError("No D/K/S members were requested")
    return merged


def build_test_frame(results: Path = RESULTS, dks_method: str = DKS_TEST_METHOD) -> pd.DataFrame:
    """Assemble the frozen ten test features without organism identity."""
    reason_rows = pd.DataFrame(
        read_jsonl(results / "qwen_reason_v1/test/generations.jsonl")
    )[["dataset", "index", "label"]].drop_duplicates()
    reason_rows["index"] = reason_rows["index"].astype(str)
    frame = reason_rows

    dks = load_member_predictions(
        results,
        dks_method,
        "test",
        {
            "details3072_logit": "details_logit",
            "known3072_logit": "known_logit",
            "scrutiny3072_logit": "scrutiny_logit",
        },
    )
    frame = join_feature(frame, dks)
    for prefix, method in (
        ("fast", "qwen_reason_nothink_truth_hidden_ensemble_v1"),
        ("reason", "qwen_reason_v1"),
    ):
        frame = join_feature(frame, load_rating_split(results, method, prefix, "test"))

    phoenix = pd.DataFrame(read_jsonl(PHOENIX_TEST))[[
        "dataset", "index", "empty_margin_score", "reasoning_margin_score"
    ]].rename(columns={
        "empty_margin_score": "phoenix_empty",
        "reasoning_margin_score": "phoenix_reason",
    })
    frame = join_feature(frame, phoenix)
    for path, feature in (
        (GRPO_EMPTY_TEST, "grpo_empty"),
        (GRPO_REASON_TEST, "grpo_reason"),
    ):
        scores = pd.read_csv(path, usecols=["dataset", "index", "score"]).rename(
            columns={"score": feature}
        )
        frame = join_feature(frame, scores)
    frame["scenario"] = frame["dataset"].map(scenario_from_dataset)
    if (
        len(frame) != len(reason_rows)
        or frame["dataset"].nunique() != 21
        or frame.isna().any().any()
    ):
        raise ValueError(f"Incomplete locked test feature frame: shape={frame.shape}")
    return frame


def fit_locked_stack(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[Any, pd.Series]:
    """Fit the frozen C=0.1 stack and return untouched test probabilities."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2_000,
            random_state=27,
        ),
    )
    model.fit(validation[features], validation["label"])
    return model, pd.Series(model.predict_proba(test[features])[:, 1], index=test.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dks-method", default=DKS_TEST_METHOD)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS / "continuous_hybrid_locked_test_v1/test",
    )
    args = parser.parse_args()

    validation = build_feature_frame()
    features = feature_sets(validation)["compact_continuous"]
    test = build_test_frame(dks_method=args.dks_method)
    model, scores = fit_locked_stack(validation, test, features)
    scored = pd.DataFrame({
        "dataset": test["dataset"], "label": test["label"], "score": scores
    })
    scenarios = {
        name: macro_metrics(scored[test["scenario"] == name], 0.5)
        for name in sorted(test["scenario"].unique())
    }
    result = {
        "method": "continuous_hybrid_locked_test_v1",
        "split": "test",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": len(test),
        "features": features,
        "training_split": "validation",
        "regularization_c": 0.1,
        "threshold": 0.5,
        "selection_rule": "frozen after grouped validation screen; no test alternatives",
        "metrics": macro_metrics(scored, 0.5),
        "scenarios": scenarios,
        "model_steps": [name for name, _ in model.steps],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "dataset": test["dataset"],
        "index": test["index"],
        "label": test["label"],
        "score": scores,
        "deceptive": scores >= 0.5,
    }).to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
