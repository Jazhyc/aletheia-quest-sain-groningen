#!/usr/bin/env python3
"""Refit a locked reasoning n-gram configuration on all public varied rows."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reasoning_ngram_intent.train import (  # noqa: E402
    Candidate,
    THRESHOLD,
    fit_model,
    load_reasoning_split,
    top_ngrams,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-artifact", type=Path, required=True)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=ROOT / "dev_splits",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = joblib.load(args.selected_artifact)
    candidate = Candidate(**selected["candidate"])
    split_frames = [
        load_reasoning_split(split, args.splits_dir.resolve())
        for split in ("train", "validation", "test")
    ]
    full = pd.concat(split_frames, ignore_index=True)
    keys = full[["dataset", "index"]].astype(str)
    if keys.duplicated().any():
        raise ValueError("public split rows overlap")
    if full["reasoning"].eq("").any():
        raise ValueError("varied-deception reasoning unexpectedly contains empty rows")

    vectorizer, classifier = fit_model(full, candidate)
    artifact = {
        "candidate": dataclasses.asdict(candidate),
        "threshold": THRESHOLD,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "provenance": {
            "fit": "all public varied-deception train/validation/test rows",
            "selection_artifact": str(args.selected_artifact.resolve()),
            "selection_artifact_sha256": sha256(args.selected_artifact),
            "selection_and_threshold_locked_before_full_refit": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifact": str(args.output.resolve()),
        "artifact_sha256": sha256(args.output),
        "candidate": dataclasses.asdict(candidate),
        "threshold": THRESHOLD,
        "rows": len(full),
        "datasets": int(full["dataset"].nunique()),
        "positives": int(full["label"].sum()),
        "empty_reasoning": int(full["reasoning"].eq("").sum()),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "top_ngrams": top_ngrams(vectorizer, classifier),
        "provenance": artifact["provenance"],
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
