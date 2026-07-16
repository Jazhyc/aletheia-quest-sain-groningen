#!/usr/bin/env python3
"""Stack cached continuous judge features with grouped out-of-fold evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import macro_metrics


RESULTS = ROOT / "results" / "blackbox"
SELF_QUERY = RESULTS / "all_organisms_ndif_whole_output_self_query_validation_v1/generations.jsonl"
PHOENIX_MARGINS = RESULTS / (
    "qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/"
    "validation_continuous_margin_v1/generations.jsonl"
)
GRPO_EMPTY = RESULTS / (
    "qwen_grpo_lora_r16_binary_t12_muonlr3e5_full_v2_logits_"
    "empty_reasoning_spaceprefix/validation/predictions.csv"
)
GRPO_REASON = RESULTS / (
    "qwen_grpo_lora_r16_binary_t12_muonlr3e5_full_v2_logits_"
    "generated_reasoning/validation/predictions.csv"
)

LOGIT_RUNS = {
    "details_logit": "qwen_reason_details3072_speed_logit_v1",
    "known_logit": "qwen_reason_known3072_logit_v1",
    "scrutiny_logit": "qwen_reason_scrutiny3072_logit_v1",
}
RATING_RUNS = {
    "heavy": "qwen_reason_ensemble_dks_member4096_v1",
    "fast": "qwen_reason_nothink_truth_hidden_ensemble_v1",
    "reason": "qwen_reason_v1",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON lines."""
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def join_feature(frame: pd.DataFrame, feature: pd.DataFrame) -> pd.DataFrame:
    """Join exactly one feature row to every dataset/index key."""
    left = frame.copy()
    right = feature.copy()
    left["index"] = left["index"].astype(str)
    right["index"] = right["index"].astype(str)
    merged = left.merge(right, on=["dataset", "index"], validate="one_to_one")
    if len(merged) != len(left):
        raise ValueError(f"Feature join lost rows: {len(merged)} != {len(left)}")
    return merged


def load_prediction_run(results: Path, method: str, feature: str) -> pd.DataFrame:
    """Load per-dataset prediction CSVs from a black-box judge run."""
    run_dir = results / method / "validation"
    result = json.loads((run_dir / "result.json").read_text())
    pieces = []
    for dataset_result in result["datasets"]:
        path = Path(dataset_result["predictions_path"])
        if not path.exists():
            path = run_dir / "predictions" / path.name
        part = pd.read_csv(path, usecols=["index", "score"])
        part["dataset"] = dataset_result["dataset"]
        pieces.append(part.rename(columns={"score": feature}))
    return pd.concat(pieces, ignore_index=True)


def load_rating_run(results: Path, method: str, prefix: str) -> pd.DataFrame:
    """Pivot member-level generated ratings into normalized score columns."""
    rows = pd.DataFrame(read_jsonl(results / method / "validation/generations.jsonl"))
    rows["index"] = rows["index"].astype(str)
    rows["member"] = rows.get("ensemble_member", "single")
    rows["rating"] = rows["rating"].fillna(1).astype(float)
    pivot = rows.pivot(
        index=["dataset", "index"], columns="member", values="rating"
    ).reset_index()
    pivot.columns = [
        "dataset",
        "index",
        *(f"{prefix}_{column}" for column in pivot.columns[2:]),
    ]
    rating_columns = [column for column in pivot if column.startswith(f"{prefix}_")]
    pivot[rating_columns] = (pivot[rating_columns] - 1.0) / 6.0
    return pivot


def organism_key(model: str, lora: Any) -> str:
    """Build the grouping key shared by instructed and varied datasets."""
    suffix = "base" if pd.isna(lora) or not lora else str(lora)
    return f"{model}::{suffix}"


def family_from_model(model: str) -> str:
    """Map model identifiers to the three public organism families."""
    lowered = model.casefold()
    if "qwen" in lowered:
        return "Qwen"
    if "gemma" in lowered:
        return "Gemma"
    if "nemotron" in lowered:
        return "Nemotron"
    return model


def scenario_from_dataset(dataset: str) -> str:
    """Map a dataset name to its deception scenario."""
    return "varied-deception" if "varied-deception" in dataset else "instructed-deception"


def load_organism_margins(path: Path) -> pd.DataFrame:
    """Pivot the two Qwen neutral-audit margins; leave non-Qwen rows neutral later."""
    rows = pd.DataFrame(read_jsonl(path))
    required = {"dataset", "index", "variant", "false_score"}
    if not required.issubset(rows.columns):
        raise ValueError(f"{path} lacks organism margin fields {sorted(required - set(rows.columns))}")
    rows["index"] = rows["index"].astype(str)
    pivot = rows.pivot(
        index=["dataset", "index"], columns="variant", values="false_score"
    ).reset_index()
    pivot.columns = [
        "dataset",
        "index",
        *(f"organism_margin_{column}" for column in pivot.columns[2:]),
    ]
    return pivot


def build_feature_frame(
    *,
    results: Path = RESULTS,
    self_query_path: Path = SELF_QUERY,
    phoenix_path: Path = PHOENIX_MARGINS,
    grpo_empty_path: Path = GRPO_EMPTY,
    grpo_reason_path: Path = GRPO_REASON,
    organism_margins: Path | None = None,
) -> pd.DataFrame:
    """Assemble the frozen row-level feature bank."""
    base = pd.DataFrame(read_jsonl(self_query_path))
    base["index"] = base["index"].astype(str)
    base["organism"] = [
        organism_key(model, lora) for model, lora in zip(base["model"], base["lora"], strict=True)
    ]
    base["family"] = base["model"].map(family_from_model)
    base["scenario"] = base["dataset"].map(scenario_from_dataset)
    base["organism_reject"] = base["deception_score"].astype(float)
    frame = base[[
        "dataset", "index", "label", "organism", "family", "scenario", "organism_reject"
    ]].copy()

    for feature, method in LOGIT_RUNS.items():
        frame = join_feature(frame, load_prediction_run(results, method, feature))
    for prefix, method in RATING_RUNS.items():
        frame = join_feature(frame, load_rating_run(results, method, prefix))

    phoenix = pd.DataFrame(read_jsonl(phoenix_path))[[
        "dataset", "index", "empty_margin_score", "reasoning_margin_score"
    ]].rename(columns={
        "empty_margin_score": "phoenix_empty",
        "reasoning_margin_score": "phoenix_reason",
    })
    frame = join_feature(frame, phoenix)
    for path, feature in ((grpo_empty_path, "grpo_empty"), (grpo_reason_path, "grpo_reason")):
        scores = pd.read_csv(path, usecols=["dataset", "index", "score"]).rename(
            columns={"score": feature}
        )
        frame = join_feature(frame, scores)

    if organism_margins is not None:
        margins = load_organism_margins(organism_margins)
        frame = frame.merge(margins, on=["dataset", "index"], how="left", validate="one_to_one")
        margin_columns = [column for column in frame if column.startswith("organism_margin_")]
        frame[margin_columns] = frame[margin_columns].fillna(0.5)

    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Feature frame contains missing values in {missing}")
    return frame


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return the exploratory cached sets and frozen organism-margin extension."""
    heavy = sorted(column for column in frame if column.startswith("heavy_"))
    compact = [
        "details_logit", "known_logit", "scrutiny_logit",
        "fast_hidden_contradiction", "fast_truth_value_guard", "reason_single",
        "phoenix_empty", "phoenix_reason", "grpo_empty", "grpo_reason",
    ]
    sets = {
        "heavy_ratings": heavy,
        "compact_continuous": compact,
        "compact_plus_binary_organism": [*compact, "organism_reject"],
        "all_judges": [*compact, *heavy],
        "all_plus_binary_organism": [*compact, *heavy, "organism_reject"],
    }
    margin_columns = sorted(column for column in frame if column.startswith("organism_margin_"))
    if margin_columns:
        sets["compact_plus_continuous_organism"] = [*compact, *margin_columns]
    return sets


def grouped_oof_scores(
    frame: pd.DataFrame,
    features: list[str],
    group_column: str,
    *,
    regularization_c: float = 0.1,
) -> np.ndarray:
    """Fit on every other group and return one probability for every held-out row."""
    scores = np.full(len(frame), np.nan, dtype=float)
    for group in sorted(frame[group_column].unique()):
        test = frame[group_column] == group
        train = ~test
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=regularization_c,
                class_weight="balanced",
                max_iter=2_000,
                random_state=27,
            ),
        )
        model.fit(frame.loc[train, features], frame.loc[train, "label"])
        scores[test.to_numpy()] = model.predict_proba(frame.loc[test, features])[:, 1]
    if np.isnan(scores).any():
        raise AssertionError(f"{group_column} OOF evaluation left rows unscored")
    return scores


def metric_input(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"dataset": frame["dataset"], "label": frame["label"], "score": scores})


def breakdown(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    """Report overall metrics plus scenario and family controls."""
    scored = frame.copy()
    scored["score"] = scores

    def groups(column: str) -> dict[str, Any]:
        return {
            str(name): {
                "rows": len(group),
                "metrics": macro_metrics(metric_input(group, group["score"].to_numpy()), 0.5),
            }
            for name, group in scored.groupby(column, sort=True)
        }

    return {
        "overall": macro_metrics(metric_input(scored, scores), 0.5),
        "scenarios": groups("scenario"),
        "families": groups("family"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--organism-margins", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS / "continuous_hybrid_validation_v1",
    )
    args = parser.parse_args()

    frame = build_feature_frame(organism_margins=args.organism_margins)
    evaluations: dict[str, Any] = {}
    prediction_frame = frame[[
        "dataset", "index", "label", "organism", "family", "scenario"
    ]].copy()
    for name, features in feature_sets(frame).items():
        evaluations[name] = {"features": features, "groupings": {}}
        for grouping in ("dataset", "organism"):
            scores = grouped_oof_scores(frame, features, grouping)
            key = f"{name}__leave_one_{grouping}_out"
            prediction_frame[key] = scores
            evaluations[name]["groupings"][grouping] = breakdown(frame, scores)

    result = {
        "method": args.output_dir.name,
        "split": "validation",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": len(frame),
        "regularization_c": 0.1,
        "threshold": 0.5,
        "organism_margins": None if args.organism_margins is None else str(args.organism_margins),
        "notes": {
            "cached_screen": "exploratory feature selection on validation",
            "required_groupings": ["leave-one-dataset-unit-out", "leave-one-organism-out"],
            "identity_features": False,
            "non_qwen_continuous_organism_default": 0.5,
        },
        "evaluations": evaluations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
