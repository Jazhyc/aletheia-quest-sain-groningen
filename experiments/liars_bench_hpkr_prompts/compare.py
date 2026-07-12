#!/usr/bin/env python3
"""Compare aligned and Heretic HP-KR prompt-sweep results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results/blackbox/liars_bench_hpkr_epistemic_prompt_sweep_v1"
CONDITIONS = ("base_qwen", "heretic_v2")


def main() -> None:
    results = {name: json.loads((OUTPUT_DIR / name / "result.json").read_text()) for name in CONDITIONS}
    prompts = results[CONDITIONS[0]]["prompts"]
    comparison = {"conditions": {}, "paired_changes": {}}
    for condition, result in results.items():
        comparison["conditions"][condition] = {
            prompt: summary for prompt, summary in result["prompts"].items()
        }
    for prompt in prompts:
        base = pd.read_json(OUTPUT_DIR / "base_qwen" / f"{prompt}.jsonl", lines=True)
        heretic = pd.read_json(OUTPUT_DIR / "heretic_v2" / f"{prompt}.jsonl", lines=True)
        merged = base[["index", "label", "score"]].merge(
            heretic[["index", "score"]], on="index", suffixes=("_base", "_heretic"), validate="one_to_one"
        )
        comparison["paired_changes"][prompt] = {
            "base_0_heretic_1": int(((merged["score_base"] == 0) & (merged["score_heretic"] == 1)).sum()),
            "base_1_heretic_0": int(((merged["score_base"] == 1) & (merged["score_heretic"] == 0)).sum()),
            "deceptive_base_0_heretic_1": int(((merged["label"] == 1) & (merged["score_base"] == 0) & (merged["score_heretic"] == 1)).sum()),
            "honest_base_0_heretic_1": int(((merged["label"] == 0) & (merged["score_base"] == 0) & (merged["score_heretic"] == 1)).sum()),
        }
    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
