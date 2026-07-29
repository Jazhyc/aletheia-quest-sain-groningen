#!/usr/bin/env python3
"""Score the current structural Q397 direct baseline on the frozen OOD sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_adapter_reasoning_ndif import run as reasoning_ndif  # noqa: E402
from experiments.phoenix_adapter_validation_ndif.run import (  # noqa: E402
    MODEL_ID,
    binary_token_ids,
    hosted_metadata,
    load_credentials,
)
from experiments.q397_reasoning_ood.run_ndif import (  # noqa: E402
    ADAPTER,
    DEFAULT_FROZEN_SAMPLE,
    DEFAULT_OUTPUT_DIR,
    load_records,
)
from experiments.specialist_necessity_ndif.run import prompt_builder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-sample",
        type=Path,
        default=DEFAULT_FROZEN_SAMPLE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-batches-per-session", type=int, default=0)
    return parser.parse_args()


def structural_prompts(
    records: pd.DataFrame,
    tokenizer: Any,
) -> list[str]:
    builder = prompt_builder("general")
    return [
        builder(messages, tokenizer)
        for messages in records["messages"]
    ]


def main() -> None:
    args = parse_args()
    load_credentials()
    records = load_records(args.frozen_sample.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from nnsight import LanguageModel

    metadata = hosted_metadata(ADAPTER.repo_id)
    model = LanguageModel(MODEL_ID, peft=ADAPTER.repo_id)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    scores, audit = reasoning_ndif.score_prompt_sets(
        model,
        tokenizer,
        binary_token_ids(tokenizer),
        {"structural_direct": structural_prompts(records, tokenizer)},
        batches_per_session=args.score_batches_per_session,
    )
    output = records.drop(columns="messages").copy()
    output["structural_direct_score"] = scores["structural_direct"]
    output.to_json(
        output_dir / "structural_direct_scores.jsonl",
        orient="records",
        lines=True,
        double_precision=15,
    )
    report = {
        "rows": len(records),
        "adapter": {
            "name": ADAPTER.name,
            "repo_id": ADAPTER.repo_id,
            "hosted": metadata,
        },
        "specialist_prompts_used": False,
        "renderer": "phoenix_4_structural",
        "scoring": audit,
    }
    (output_dir / "structural_direct_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
