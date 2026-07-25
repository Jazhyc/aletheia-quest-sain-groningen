#!/usr/bin/env python3
"""Smoke-test Phoenix 4.1's routed base-Qwen margin on NDIF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.phoenix_wright_v4 import (
    binary_token_ids,
    build_epistemic_empty_prompt,
    is_knowledge_report_request,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/blackbox/phoenix_v4_1_epistemic_ndif_smoke.json",
    )
    args = parser.parse_args()
    if args.limit < 1 or args.batch_size < 1:
        raise ValueError("limit and batch size must be positive")
    if not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required")
    os.environ.setdefault("NDIF_HOST", "https://aletheias.api.ndif.us")

    from datasets import load_dataset
    try:
        from nnsight import LanguageModel
    except ImportError:
        from nnsight.modeling.language import LanguageModel

    dataset = load_dataset(
        "Cadenza-Labs/liars-bench",
        data_dir="harm-pressure-knowledge-report",
        split="test",
    ).select(range(args.limit))
    messages = list(dataset["messages"])
    if not all(is_knowledge_report_request(value) for value in messages):
        raise RuntimeError("the frozen HP-KR router did not cover every smoke row")

    model = LanguageModel("Qwen/Qwen3.5-9B")
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = list(binary_token_ids(tokenizer))
    prompts = [
        build_epistemic_empty_prompt(value, tokenizer)
        for value in messages
    ]

    scores = []
    started = time.perf_counter()
    with model.session(remote=True):
        saved = []
        for start in range(0, len(prompts), args.batch_size):
            batch = tokenizer(
                prompts[start:start + args.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            with model.trace({
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "logits_to_keep": 1,
            }):
                logits = model.output.logits[:, -1, label_ids].float()
                saved.append(
                    torch.softmax(logits, dim=-1)[:, 1].detach().cpu()
                )
        remote_scores = torch.cat(saved).save()
    scores = remote_scores.float().numpy()
    elapsed = time.perf_counter() - started

    result = {
        "rows": len(scores),
        "route_coverage": len(scores),
        "unique_scores": int(np.unique(scores).size),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "elapsed_seconds": elapsed,
        "model_path": "base Qwen/Qwen3.5-9B without PEFT",
        "margin_path": "empty reasoning summary then Prediction: 0/1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
