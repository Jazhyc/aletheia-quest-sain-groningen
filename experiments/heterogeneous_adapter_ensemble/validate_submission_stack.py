#!/usr/bin/env python3
"""Verify that the packaged Phoenix 3.0 stack reproduces the frozen test row."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))

from experiments.heterogeneous_adapter_ensemble.analyze_ngram_stack import attach_text
from experiments.pid_specialist_ensemble.analyze_ensemble import load_member_frame, metrics
from experiments.text_probe.run_text_probe import load_split


def main() -> None:
    members = [
        (
            "deception",
            ROOT / "results/blackbox/qwen9b_pid_specialist_material_rank1_v1/"
            "test_heterogeneous_objectives_v1/generations.jsonl",
        ),
        (
            "resolved_intent",
            ROOT / "results/blackbox/qwen9b_heterogeneous_resolved_intent_rank1_v1/"
            "test_heterogeneous_objectives_v1/generations.jsonl",
        ),
    ]
    frame = load_member_frame(members).reset_index(drop=True)
    text = load_split(
        "test", ROOT / "dev_splits", max_context_chars=3000
    ).frame
    frame = attach_text(frame, text)
    artifact = joblib.load(ROOT / "submission/phoenix_wright_v3_ngram.joblib")
    text_features = artifact["vectorizer"].transform(frame["text"])
    member_features = frame[list(artifact["member_names"])].to_numpy(dtype=float)
    features = hstack(
        [text_features, csr_matrix(member_features)], format="csr"
    )
    scores = artifact["classifier"].predict_proba(features)[:, 1]
    actual = metrics(frame, scores)
    expected = {
        "balanced_accuracy": 0.9238095238095237,
        "auroc": 0.9549999999999998,
        "recall": 0.8857142857142856,
        "fpr": 0.0380952380952381,
    }
    for key in ("balanced_accuracy", "auroc", "recall", "fpr"):
        if abs(actual[key] - expected[key]) > 1e-12:
            raise AssertionError(f"{key}: {actual[key]} != {expected[key]}")
    print(json.dumps({
        "rows": len(frame),
        "balanced_accuracy": actual["balanced_accuracy"],
        "auroc": actual["auroc"],
        "recall": actual["recall"],
        "fpr": actual["fpr"],
        "replay": "exact",
    }, indent=2))


if __name__ == "__main__":
    main()
