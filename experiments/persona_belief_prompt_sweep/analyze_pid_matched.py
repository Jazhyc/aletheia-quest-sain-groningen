#!/usr/bin/env python3
"""Analyze the PI student weight-by-prompt persona ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.persona_belief_prompt_sweep.analyze import (
    ROOT,
    is_persona_roleplay,
    load_examples_for_labels,
    load_label_subset,
    load_split_config,
    summarize_member,
)


CONDITIONS = ("control_prompt", "persona_prompt")


def condition_result(adapter: Path, run_name: str, condition: str) -> Path:
    """Return the evaluator result path for one adapter/prompt condition."""
    return adapter.parent / run_name / condition / "result.json"


def load_condition_frame(
    result_path: Path,
    persona_frames: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    """Replace the template scores with direct margins from one condition."""
    generations = pd.read_json(
        result_path.parent / "generations.jsonl",
        lines=True,
    )
    generations["_key"] = generations["index"].astype(str)
    scores = generations.set_index(["dataset", "_key"])[
        "direct_margin_score"
    ]
    frames: list[pd.DataFrame] = []
    for template in persona_frames:
        frame = template.copy()
        dataset = str(frame["dataset"].iloc[0])
        frame["score"] = [
            float(scores.loc[(dataset, str(index))])
            for index in frame["_key"]
        ]
        frames.append(frame)
    return frames


def load_persona_templates(split_config: Path) -> list[pd.DataFrame]:
    """Load labels plus the frozen system-only persona diagnostic once."""
    frames: list[pd.DataFrame] = []
    for config in load_split_config(split_config, ROOT):
        labels = load_label_subset(config).copy()
        labels["_key"] = labels["index"].astype(str)
        examples = load_examples_for_labels(config.name, labels, None)
        persona_by_index = {
            str(example["index"]): is_persona_roleplay(example["messages"])
            for example in examples
        }
        frame = labels[["_key", "label"]].copy()
        frame["dataset"] = config.name
        frame["persona"] = (
            frame["_key"].map(persona_by_index).fillna(False).astype(bool)
        )
        frames.append(frame)
    return frames


def analyze(
    control_adapter: Path,
    matched_adapter: Path,
    run_name: str,
    split_config: Path,
) -> dict[str, Any]:
    """Build the two-adapter by two-prompt AUROC report."""
    template_frames = load_persona_templates(split_config)
    adapters = {
        "existing_weights": control_adapter,
        "persona_matched_weights": matched_adapter,
    }
    cells: dict[str, Any] = {}
    for weight_name, adapter in adapters.items():
        for condition in CONDITIONS:
            result_path = condition_result(adapter, run_name, condition)
            summary = summarize_member(
                load_condition_frame(result_path, template_frames)
            )
            cells[f"{weight_name}/{condition}"] = summary

    baseline = cells["existing_weights/control_prompt"][
        "macro_per_dataset_auroc"
    ]
    for summary in cells.values():
        summary["delta_vs_existing_control"] = (
            float(summary["macro_per_dataset_auroc"] - baseline)
        )
    return {
        "selection_metric": "mean per-dataset direct-margin AUROC",
        "selection_uses_thresholded_metrics": False,
        "run_name": run_name,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-adapter", type=Path, required=True)
    parser.add_argument("--matched-adapter", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=ROOT / "dev_splits" / "dry.validation.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = analyze(
        args.control_adapter.resolve(),
        args.matched_adapter.resolve(),
        args.run_name,
        args.split_config.resolve(),
    )
    print("weights/persona prompt                         macro_AUROC   delta")
    for name, summary in report["cells"].items():
        print(
            f"{name:<48} "
            f"{summary['macro_per_dataset_auroc']:.6f} "
            f"{summary['delta_vs_existing_control']:+.6f}"
        )
    output = args.output or (
        args.matched_adapter.resolve().parent
        / args.run_name
        / "pid_persona_matched_analysis.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
