#!/usr/bin/env python3
"""Audit literal value disagreements between saved judge replies and outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.blackbox.judge_output_consistency_rules import (  # noqa: E402
    judge_output_value_disagreement,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import (  # noqa: E402
    load_split_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    return parser.parse_args()


def main() -> None:
    from datasets import disable_progress_bar, load_dataset

    args = parse_args()
    disable_progress_bar()
    generations = {
        (record["dataset"], record["index"]): record
        for line in args.generations.read_text().splitlines()
        if (record := json.loads(line))
    }
    counts: Counter[str] = Counter()
    flagged: list[dict] = []
    for config in load_split_config(args.splits_dir / f"dry.{args.split}.yaml", ROOT):
        dataset = load_dataset(config.name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        for row in dataset:
            record = generations.get((config.name, row["index"]))
            if record is None or record["prediction"] != 0:
                continue
            flag = judge_output_value_disagreement(row["messages"], record["generation"])
            if not flag:
                continue
            label = int(record["label"])
            counts["flags"] += 1
            counts["true_flags"] += label == 1
            counts["false_flags"] += label == 0
            flagged.append({
                "dataset": config.name,
                "index": row["index"],
                "label": label,
                "rule": "physical_state",
                "output": next(
                    message["content"]
                    for message in reversed(row["messages"])
                    if message["role"] == "assistant"
                ),
                "judge_reply": record["generation"],
            })
    print(json.dumps({"split": args.split, "counts": counts, "flagged": flagged}, indent=2))


if __name__ == "__main__":
    main()
