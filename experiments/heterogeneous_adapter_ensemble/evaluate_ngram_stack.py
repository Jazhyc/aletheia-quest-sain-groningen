#!/usr/bin/env python3
"""Apply the frozen train-fitted heterogeneous n-gram stack to one split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.heterogeneous_adapter_ensemble.analyze_ngram_stack import (
    attach_text,
    fit_scores,
)
from experiments.pid_specialist_ensemble.analyze_ensemble import (
    comparison_counts,
    load_member_frame,
    load_selection_keys,
    metrics,
    parse_member,
)
from experiments.text_probe.run_text_probe import load_split


def build_report(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    member_names: list[str],
    *,
    split: str,
    excluded_rows: int,
    max_context_chars: int,
) -> dict[str, Any]:
    """Fit the frozen stack on train and report one untouched evaluation split."""
    scores, model_metadata = fit_scores(train, evaluation, member_names)
    labels = evaluation["label"].to_numpy(dtype=int)
    deception_scores = evaluation["deception"].to_numpy(dtype=float)
    adapter_scores = scores["adapter_only"]
    return {
        "protocol": {
            "fit_split": "train",
            "evaluation_split": split,
            "text_view": "output_context_without_reasoning",
            "max_context_chars": max_context_chars,
            "vectorizer": {
                "analyzer": "word",
                "ngram_range": [1, 2],
                "min_df": 3,
                "max_df": 0.98,
                "max_features": 20_000,
                "sublinear_tf": True,
            },
            "classifier": {
                "type": "logistic_regression",
                "c": 1.0,
                "threshold": 0.5,
            },
            "member_order": member_names,
            "base_training_rows_excluded": excluded_rows,
            "meta_training_rows": len(train),
        },
        split: {
            name: metrics(evaluation, values) for name, values in scores.items()
        },
        f"{split}_comparison": {
            name: {
                "versus_deception": comparison_counts(
                    labels, deception_scores, values
                ),
                "versus_adapter_logistic": comparison_counts(
                    labels, adapter_scores, values
                ),
            }
            for name, values in scores.items()
        },
        "model": model_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-member", action="append", required=True)
    parser.add_argument("--evaluation-member", action="append", required=True)
    parser.add_argument(
        "--evaluation-split", choices=["validation", "test"], required=True
    )
    parser.add_argument(
        "--selection-manifest", action="append", type=Path, required=True
    )
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--max-context-chars", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_specs = [parse_member(value) for value in args.train_member]
    evaluation_specs = [parse_member(value) for value in args.evaluation_member]
    member_names = [name for name, _ in train_specs]
    if [name for name, _ in evaluation_specs] != member_names:
        raise ValueError("train and evaluation member names/order must match")

    train = load_member_frame(train_specs).reset_index(drop=True)
    evaluation = load_member_frame(evaluation_specs).reset_index(drop=True)
    selected = load_selection_keys(
        [path.resolve() for path in args.selection_manifest]
    )
    leaked = pd.Series(
        [
            (str(dataset), str(index)) in selected
            for dataset, index in zip(
                train["dataset"], train["index"], strict=True
            )
        ]
    )
    train = train.loc[~leaked].reset_index(drop=True)

    splits_dir = args.splits_dir.resolve()
    train_text = load_split(
        "train", splits_dir, max_context_chars=args.max_context_chars
    ).frame
    evaluation_text = load_split(
        args.evaluation_split,
        splits_dir,
        max_context_chars=args.max_context_chars,
    ).frame
    train = attach_text(train, train_text)
    evaluation = attach_text(evaluation, evaluation_text)
    report = build_report(
        train,
        evaluation,
        member_names,
        split=args.evaluation_split,
        excluded_rows=int(leaked.sum()),
        max_context_chars=args.max_context_chars,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
