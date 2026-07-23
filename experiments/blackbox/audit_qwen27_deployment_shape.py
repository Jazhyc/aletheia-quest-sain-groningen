#!/usr/bin/env python3
"""Measure Qwen-27B D/K/S prompt shapes without running model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.blackbox.run_judge import (
    build_prompt,
    format_dialogue,
    load_examples_for_labels,
    load_label_subset,
    load_split_config,
)


def summarize(values: list[int]) -> dict[str, int]:
    array = np.asarray(values, dtype=int)
    if not len(array):
        return {"n": 0, "median": 0, "p95": 0, "max": 0}
    return {
        "n": int(len(array)),
        "median": int(np.median(array)),
        "p95": int(np.percentile(array, 95)),
        "max": int(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
    )
    parser.add_argument(
        "--judge-config",
        type=Path,
        default=(
            ROOT
            / "results/blackbox/qwen27b_reason_ensemble_dks_member4096_v1/validation/config.yaml"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.judge_config)
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.judge.model))
    members = [str(member.prompt) for member in cfg.ensemble.members]
    max_chars = int(cfg.judge.max_prompt_chars)
    truncation = str(cfg.judge.get("context_truncation", "tail"))

    result: dict[str, Any] = {
        "model": str(cfg.judge.model),
        "members": len(members),
        "max_prompt_chars": max_chars,
        "context_truncation": truncation,
        "splits": {},
    }
    all_lengths: list[int] = []
    all_rows = all_truncated = 0
    for split in args.splits:
        split_lengths: list[int] = []
        split_rows = split_truncated = 0
        units: dict[str, dict[str, Any]] = {}
        configs = load_split_config(args.splits_dir / f"dry.{split}.yaml", ROOT)
        for dataset_cfg in configs:
            labels = load_label_subset(dataset_cfg)
            examples = load_examples_for_labels(dataset_cfg.name, labels, None)
            unit_lengths: list[int] = []
            unit_truncated = 0
            for example in examples:
                raw_context, _ = format_dialogue(example["messages"], 10**9)
                unit_truncated += int(len(raw_context) > max_chars)
                for prompt in members:
                    rendered = build_prompt(
                        example["messages"],
                        prompt,
                        max_chars,
                        append_prefix=None,
                        context_truncation=truncation,
                    )
                    unit_lengths.append(
                        len(tokenizer.encode(rendered, add_special_tokens=False))
                    )
            units[str(dataset_cfg.name)] = {
                "rows": len(examples),
                "prompt_evaluations": len(unit_lengths),
                "truncated_rows": unit_truncated,
                "prompt_tokens": summarize(unit_lengths),
                "prompt_tokens_over_2048": sum(v > 2048 for v in unit_lengths),
            }
            split_rows += len(examples)
            split_truncated += unit_truncated
            split_lengths.extend(unit_lengths)
        result["splits"][split] = {
            "rows": split_rows,
            "prompt_evaluations": len(split_lengths),
            "truncated_rows": split_truncated,
            "prompt_tokens": summarize(split_lengths),
            "prompt_tokens_over_2048": sum(v > 2048 for v in split_lengths),
            "maximum_unit_rows": max((v["rows"] for v in units.values()), default=0),
            "maximum_unit_prompt_evaluations": max(
                (v["prompt_evaluations"] for v in units.values()), default=0
            ),
            "units": units,
        }
        all_rows += split_rows
        all_truncated += split_truncated
        all_lengths.extend(split_lengths)
    result["all"] = {
        "rows": all_rows,
        "prompt_evaluations": len(all_lengths),
        "truncated_rows": all_truncated,
        "prompt_tokens": summarize(all_lengths),
        "prompt_tokens_over_2048": sum(v > 2048 for v in all_lengths),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all": result["all"], "splits": {
        split: {key: value for key, value in values.items() if key != "units"}
        for split, values in result["splits"].items()
    }}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
