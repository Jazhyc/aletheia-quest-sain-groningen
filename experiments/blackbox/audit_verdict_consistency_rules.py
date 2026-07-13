#!/usr/bin/env python3
"""Audit verdict-consistency rule coverage on a labeled local split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.blackbox.verdict_consistency_rules import (  # noqa: E402
    has_dishonest_instruction,
    verdict_explanation_conflict,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import (  # noqa: E402
    load_labels,
    load_split_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-text", action="store_true")
    return parser.parse_args()


def family_name(dataset: str) -> str:
    if "Nemotron" in dataset:
        return "nemotron"
    if "gemma" in dataset:
        return "gemma"
    return "qwen"


def metric_record(counts: Counter[str]) -> dict[str, int | float | None]:
    flags = counts["flags"]
    positives = counts["positives"]
    return {
        **dict(counts),
        "flag_precision": counts["true_flags"] / flags if flags else None,
        "positive_coverage": counts["true_flags"] / positives if positives else None,
    }


def main() -> None:
    from datasets import disable_progress_bar, load_dataset

    args = parse_args()
    disable_progress_bar()
    total: Counter[str] = Counter()
    groups: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    flagged_rows: list[dict[str, Any]] = []

    split_config = args.splits_dir / f"dry.{args.split}.yaml"
    for config in load_split_config(split_config, ROOT):
        labels = load_labels(config)
        label_by_index = dict(zip(labels["index"], labels["label"], strict=True))
        dataset = load_dataset(config.name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        scenario = "varied" if "varied-deception" in config.name else "instructed"
        family = family_name(config.name)

        for row in dataset:
            index = row["index"]
            if index not in label_by_index:
                continue
            messages = [dict(message) for message in row["messages"]]
            label = int(label_by_index[index])
            intent = has_dishonest_instruction(messages)
            flag = verdict_explanation_conflict(messages)
            for counts in (total, groups[(scenario, family)]):
                counts["rows"] += 1
                counts["positives"] += label == 1
                counts["intent_contexts"] += intent
                counts["flags"] += flag
                counts["true_flags"] += flag and label == 1
                counts["false_flags"] += flag and label == 0
            if flag:
                record: dict[str, Any] = {
                    "dataset": config.name,
                    "index": index,
                    "label": label,
                    "scenario": scenario,
                    "family": family,
                }
                if args.include_text:
                    record["messages"] = messages
                flagged_rows.append(record)

    result = {
        "split": args.split,
        "overall": metric_record(total),
        "groups": {
            f"{scenario}/{family}": metric_record(counts)
            for (scenario, family), counts in sorted(groups.items())
        },
        "flagged_rows": flagged_rows,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
