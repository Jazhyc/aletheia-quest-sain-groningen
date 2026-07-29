#!/usr/bin/env python3
"""Compare fixed Q397 direct/reasoning blends across reasoning prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.q397_reasoning_router.analyze import (  # noqa: E402
    comparison,
    logit,
    sigmoid,
    summarize,
)


DEFAULT_ROOT = (
    ROOT
    / "results/blackbox/qwen9b_qwen397_tvg_soft_r16_lr5e5_ep2_v1"
    / "validation_reasoning_prompt_sweep_v1"
)
PROMPTS = ("summary_baseline", "claim_check", "balanced_audit")


def load_condition(root: Path, name: str) -> pd.DataFrame:
    path = root / name / "generations.jsonl"
    frame = pd.read_json(path, lines=True)
    required = {
        "dataset",
        "index",
        "label",
        "direct_margin_score",
        "reasoning_margin_score",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "analysis.json"

    binary = load_condition(args.root, "binary")
    direct = binary["direct_margin_score"].to_numpy(float)
    keys = binary[["dataset", "index", "label"]].copy()
    report: dict[str, Any] = {
        "protocol": {
            "selection_split": "validation",
            "test_used": False,
            "direct_weight": 0.6,
            "reasoning_weight": 0.4,
            "reasoning_token_cap": 192,
        },
        "direct": summarize(binary, direct),
        "prompts": {},
    }
    for name in PROMPTS:
        condition = load_condition(args.root, name)
        aligned = keys.merge(
            condition,
            on=["dataset", "index", "label"],
            how="left",
            validate="one_to_one",
        )
        if aligned["reasoning_margin_score"].isna().any():
            raise ValueError(f"{name}: incomplete join")
        reasoning = aligned["reasoning_margin_score"].to_numpy(float)
        blend = sigmoid(
            0.6 * logit(direct)
            + 0.4 * logit(reasoning)
        )
        report["prompts"][name] = {
            "reasoning": summarize(aligned, reasoning),
            "blend": summarize(aligned, blend),
            "blend_comparison_to_direct": comparison(
                aligned,
                direct,
                blend,
            ),
            "length_capped": int(aligned["finish_reason"].eq("length").sum()),
            "generation_tokens": {
                "median": float(aligned["generation_tokens"].median()),
                "p95": float(aligned["generation_tokens"].quantile(0.95)),
                "max": int(aligned["generation_tokens"].max()),
            },
        }
    selected = max(
        report["prompts"],
        key=lambda name: report["prompts"][name]["blend"]["macro_auroc"],
    )
    report["selected_prompt"] = selected
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
