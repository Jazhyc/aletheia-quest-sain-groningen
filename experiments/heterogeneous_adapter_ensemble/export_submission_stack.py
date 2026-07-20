#!/usr/bin/env python3
"""Export the frozen heterogeneous n-gram stack for a submission notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.heterogeneous_adapter_ensemble.analyze_ngram_stack import (
    append_member_features,
    attach_text,
    make_classifier,
    make_vectorizer,
)
from experiments.pid_specialist_ensemble.analyze_ensemble import (
    balanced_cell_weights,
    load_member_frame,
    load_selection_keys,
    parse_member,
)
from experiments.text_probe.run_text_probe import load_split


def fit_submission_stack(
    train: pd.DataFrame,
    member_names: list[str],
) -> dict[str, object]:
    """Fit the exact frozen hybrid model and return its deployable components."""
    vectorizer = make_vectorizer()
    text_features = vectorizer.fit_transform(train["text"])
    member_features = train[member_names].to_numpy(dtype=float)
    classifier = make_classifier()
    classifier.fit(
        append_member_features(text_features, member_features),
        train["label"].to_numpy(dtype=int),
        sample_weight=balanced_cell_weights(train),
    )
    return {
        "format_version": 1,
        "method": "phoenix_wright_v3.0",
        "member_names": member_names,
        "max_context_chars": 3000,
        "threshold": 0.5,
        "vectorizer": vectorizer,
        "classifier": classifier,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-member", action="append", required=True)
    parser.add_argument("--selection-manifest", action="append", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--max-context-chars", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    specs = [parse_member(value) for value in args.train_member]
    member_names = [name for name, _ in specs]
    train = load_member_frame(specs).reset_index(drop=True)
    selected = load_selection_keys([path.resolve() for path in args.selection_manifest])
    leaked = pd.Series([
        (str(dataset), str(index)) in selected
        for dataset, index in zip(train["dataset"], train["index"], strict=True)
    ])
    train = train.loc[~leaked].reset_index(drop=True)
    train_text = load_split(
        "train",
        args.splits_dir.resolve(),
        max_context_chars=args.max_context_chars,
    ).frame
    train = attach_text(train, train_text)
    artifact = fit_submission_stack(train, member_names)
    artifact["max_context_chars"] = args.max_context_chars
    artifact["meta_training_rows"] = len(train)
    artifact["base_training_rows_excluded"] = int(leaked.sum())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    print(json.dumps({
        "output": args.output.as_posix(),
        "bytes": args.output.stat().st_size,
        "member_names": member_names,
        "vocabulary_size": len(artifact["vectorizer"].vocabulary_),
        "meta_training_rows": len(train),
        "base_training_rows_excluded": int(leaked.sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
