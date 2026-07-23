#!/usr/bin/env python3
"""Bundle the frozen reasoning n-gram and rank-24 meta-model for Phoenix."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import joblib


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngram-artifact", type=Path, required=True)
    parser.add_argument("--meta-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ngram = joblib.load(args.ngram_artifact)
    meta = joblib.load(args.meta_artifact)
    if ngram["candidate"] != {
        "truncation": "head_tail",
        "max_chars": 4000,
        "min_df": 8,
        "c": 0.3,
    }:
        raise ValueError(f"unexpected n-gram candidate: {ngram['candidate']}")
    if meta["feature_order"] != ["phoenix_score", "reasoning_ngram_score"]:
        raise ValueError(f"unexpected meta feature order: {meta['feature_order']}")
    if float(meta["selected_c"]) != 1.0:
        raise ValueError(f"unexpected meta regularization: {meta['selected_c']}")

    artifact = {
        "format_version": 1,
        "method": "phoenix_wright_v3.1_reasoning_ngram_logistic",
        "reasoning_max_chars": 4000,
        "reasoning_truncation": "head_tail",
        "threshold": 0.5,
        "vectorizer": ngram["vectorizer"],
        "ngram_classifier": ngram["classifier"],
        "meta_feature_order": meta["feature_order"],
        "meta_coefficient": meta["classifier"].coef_[0].tolist(),
        "meta_intercept": float(meta["classifier"].intercept_[0]),
        "provenance": {
            "ngram_artifact_sha256": sha256(args.ngram_artifact),
            "meta_artifact_sha256": sha256(args.meta_artifact),
            "ngram_fit": "public varied train rows only",
            "meta_fit": "public varied validation rows only",
            "test_used_for_fit_or_selection": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    print(f"{args.output} sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
