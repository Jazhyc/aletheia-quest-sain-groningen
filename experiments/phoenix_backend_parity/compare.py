#!/usr/bin/env python3
"""Compare matched vLLM and NDIF Phoenix backend predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "results/blackbox/phoenix_backend_parity_eunomia_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = pd.read_csv(args.root / "vllm/predictions.csv")
    right = pd.read_csv(args.root / "ndif/predictions.csv")
    if "score_migrated_adapter" in left:
        left["score"] = left["score_migrated_adapter"]
    keys = ["dataset", "index", "label", "prompt_sha256"]
    merged = left.merge(
        right,
        on=keys,
        suffixes=("_vllm", "_ndif"),
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError(
            f"row mismatch: vllm={len(left)} ndif={len(right)} matched={len(merged)}"
        )
    delta = merged["score_ndif"] - merged["score_vllm"]
    report = {
        "rows": int(len(merged)),
        "exact_equal_scores": int(
            (merged["score_vllm"] == merged["score_ndif"]).sum()
        ),
        "close_scores_at_1e_6": int(
            np.isclose(
                merged["score_vllm"],
                merged["score_ndif"],
                atol=1e-6,
                rtol=0,
            ).sum()
        ),
        "pearson": float(
            merged[["score_vllm", "score_ndif"]].corr(method="pearson").iloc[0, 1]
        ),
        "spearman": float(
            merged[["score_vllm", "score_ndif"]].corr(method="spearman").iloc[0, 1]
        ),
        "mean_absolute_difference": float(delta.abs().mean()),
        "median_absolute_difference": float(delta.abs().median()),
        "max_absolute_difference": float(delta.abs().max()),
        "mean_signed_delta_ndif_minus_vllm": float(delta.mean()),
    }
    (args.root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    merged.assign(score_delta_ndif_minus_vllm=delta).to_csv(
        args.root / "comparison.csv",
        index=False,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
