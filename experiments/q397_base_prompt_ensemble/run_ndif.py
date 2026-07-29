#!/usr/bin/env python3
"""Score the frozen explicit-digit base prompt through optimized NDIF logits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.q397_family_calibration.cache_logits import load_records
from experiments.q397_readout_adaptation.run import MODEL_ID
from experiments.phoenix_verbalizer_sweep.run_ndif_base_verbalizers import (
    CONDITIONS,
    build_direct_prompt,
    load_credentials,
    query_paired_shard,
    resolve_condition_token_ids,
    score_from_requested_logits,
)
from submission.util import build_model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "results/blackbox/q397_base_explicit_ensemble_test_base_v1"
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one remote session with Phoenix's frozen 48/32/16 batches."""
    records = load_records("test", args.limit)
    condition = next(item for item in CONDITIONS if item.name == "digits_explicit")
    conditions = (condition,)
    model = build_model(MODEL_ID)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    condition_ids, union_ids = resolve_condition_token_ids(tokenizer, conditions)
    prompts = [
        build_direct_prompt(record["messages"], tokenizer, condition)
        for record in records
    ]

    logits_by_condition, elapsed, lengths, truncated = query_paired_shard(
        model,
        tokenizer,
        {condition.name: prompts},
        union_ids,
        short_batch_size=48,
        medium_batch_size=32,
        long_batch_size=16,
        medium_threshold=600,
        long_threshold=900,
        conditions=conditions,
    )
    scores = score_from_requested_logits(
        logits_by_condition[condition.name],
        condition=condition,
        condition_ids=condition_ids[condition.name],
        union_ids=union_ids,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scores.jsonl").open("w") as handle:
        for record, prompt, score in zip(
            records, prompts, scores, strict=True
        ):
            handle.write(
                json.dumps(
                    {
                        "dataset": record["dataset"],
                        "index": record["index"],
                        "label": record["label"],
                        "score": float(score),
                        "prompt_sha256": hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    report = {
        "method": "q397_base_explicit_ensemble_test_base_v1",
        "backend": "competition_ndif",
        "split": "test",
        "rows": len(records),
        "model": MODEL_ID,
        "adapter": None,
        "condition": condition.name,
        "batch_tiers": {
            "short": 48,
            "medium": 32,
            "long": 16,
            "medium_threshold": 600,
            "long_threshold": 900,
        },
        "prompt_tokens": {
            "minimum": int(np.min(lengths)),
            "median": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95)),
            "maximum": int(np.max(lengths)),
            "truncated_rows": int(truncated),
        },
        "score_seconds": elapsed,
        "rows_per_second": len(records) / elapsed,
        "created_unix_seconds": time.time(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    load_credentials()
    run(parse_args())
