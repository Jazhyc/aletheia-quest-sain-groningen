#!/usr/bin/env python3
"""Compare pre-migration vLLM margins with corrected NDIF adapter/base scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_adapter_validation_ndif.run import paired_report


DEFAULT_RUN = ROOT / "results/blackbox/phoenix_adapter_validation_ndif_v1"
HISTORICAL = {
    "gptoss_pi": (
        ROOT
        / "results/blackbox/"
        "qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/"
        "validation_phoenix_v3_auroc_margin_sweep_v1/generations.jsonl",
        "summary",
    ),
    "gptoss_blind": (
        ROOT
        / "results/blackbox/"
        "qwen9b_blind_gptoss120b_material_reasoning_full_r16_v1/"
        "validation_blind_reasoning_full_r16_v1/generations.jsonl",
        "summary",
    ),
    "luna_pi": (
        ROOT
        / "results/blackbox/"
        "qwen9b_privileged_gpt56_luna_medium_tvg_variedonly_adamw5e5_v1/"
        "validation_continuous_margin_v1/generations.jsonl",
        "summary",
    ),
    "qwen27_soft": (
        ROOT
        / "results/blackbox/qwen9b_qwen27_tvg_binary_softonly_varied_v1/"
        "validation_qwen27_tvg_binary_soft_v1/generations.jsonl",
        "binary",
    ),
}


def aligned_scores(
    historical_path: Path,
    base_path: Path,
    adapter_path: Path,
    prompt: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    historical = pd.read_json(historical_path, lines=True)[
        ["dataset", "index", "direct_margin_score"]
    ]
    base = pd.read_csv(base_path)[
        ["dataset", "index", f"score_{prompt}"]
    ].rename(columns={f"score_{prompt}": "base_score"})
    adapter = pd.read_csv(adapter_path)[
        ["dataset", "index", f"score_{prompt}"]
    ].rename(columns={f"score_{prompt}": "adapter_score"})
    aligned = historical.merge(
        base,
        on=["dataset", "index"],
        validate="one_to_one",
    ).merge(
        adapter,
        on=["dataset", "index"],
        validate="one_to_one",
    )
    if len(aligned) != 822:
        raise ValueError(f"expected 822 aligned rows, found {len(aligned)}")
    return (
        aligned["direct_margin_score"].to_numpy(dtype=float),
        aligned["base_score"].to_numpy(dtype=float),
        aligned["adapter_score"].to_numpy(dtype=float),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    base_path = run_dir / "base_qwen_predictions.csv"
    report = {}
    for name, (historical_path, prompt) in HISTORICAL.items():
        historical, base, adapter = aligned_scores(
            historical_path,
            base_path,
            run_dir / f"{name}_predictions.csv",
            prompt,
        )
        report[name] = {
            "prompt": prompt,
            "historical_vs_corrected_base": paired_report(
                historical,
                base,
            ),
            "historical_vs_corrected_adapter": paired_report(
                historical,
                adapter,
            ),
        }
    output = run_dir / "historical_comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
