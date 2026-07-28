#!/usr/bin/env python3
"""Benchmark padding-free Phoenix direct-margin scoring through NDIF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_adapter_validation_ndif.run import (
    ADAPTERS,
    MAX_PROMPT_TOKENS,
    binary_metrics,
    binary_token_ids,
    condition_metrics,
    encode_batches,
    load_credentials,
    load_records,
    paired_report,
    prompt_templates,
    render_prompts,
)


DEFAULT_OUTPUT_DIR = (
    ROOT / "results/blackbox/phoenix_ndif_packing_v1"
)
DEFAULT_PACKED_TOKEN_BUDGET = 32_768
DEFAULT_MAX_SEQUENCES_PER_PACK = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=ROOT / "dev_splits",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=96,
        help="validation rows to score; pass 0 for all 822 rows",
    )
    parser.add_argument(
        "--packed-token-budget",
        type=int,
        default=DEFAULT_PACKED_TOKEN_BUDGET,
    )
    parser.add_argument(
        "--max-sequences-per-pack",
        type=int,
        default=DEFAULT_MAX_SEQUENCES_PER_PACK,
    )
    parser.add_argument(
        "--remote-batches-per-session",
        type=int,
        default=0,
        help="0 places every trace for one condition in one remote session",
    )
    return parser.parse_args()


def packed_position_batches(
    lengths: list[int],
    *,
    token_budget: int,
    max_sequences: int,
) -> list[list[int]]:
    """Group length-sorted rows without exceeding the flat-token budget."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if max_sequences <= 0:
        raise ValueError("max_sequences must be positive")
    if any(length <= 0 for length in lengths):
        raise ValueError("all prompt lengths must be positive")
    if any(length > token_budget for length in lengths):
        longest = max(lengths)
        raise ValueError(
            f"prompt length {longest} exceeds packed token budget "
            f"{token_budget}"
        )

    order = np.argsort(lengths)
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for raw_position in order:
        position = int(raw_position)
        length = int(lengths[position])
        if current and (
            current_tokens + length > token_budget
            or len(current) >= max_sequences
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(position)
        current_tokens += length
    if current:
        batches.append(current)
    return batches


def tokenize_prompts(tokenizer: Any, prompts: list[str]) -> list[list[int]]:
    """Apply the submission's left truncation before either batching path."""
    return [
        tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        )
        for prompt in prompts
    ]


def build_packed_batch(
    token_ids: list[list[int]],
    positions: list[int],
) -> dict[str, Any]:
    """Flatten prompts and return every boundary required by Qwen3.5."""
    from transformers import DataCollatorWithFlattening

    collator = DataCollatorWithFlattening(
        return_flash_attn_kwargs=True,
        return_seq_idx=True,
    )
    packed = dict(
        collator(
            [{"input_ids": token_ids[position]} for position in positions]
        )
    )
    packed.pop("labels", None)
    packed["logits_to_keep"] = (
        packed["cu_seq_lens_q"][1:].to(torch.long) - 1
    )
    return packed


def encode_packed_batches(
    token_ids: list[list[int]],
    *,
    token_budget: int,
    max_sequences: int,
) -> list[tuple[list[int], dict[str, Any]]]:
    lengths = [len(ids) for ids in token_ids]
    return [
        (positions, build_packed_batch(token_ids, positions))
        for positions in packed_position_batches(
            lengths,
            token_budget=token_budget,
            max_sequences=max_sequences,
        )
    ]


def score_padded(
    model: Any,
    label_ids: list[int],
    batches: list[tuple[list[int], Any]],
    *,
    rows: int,
    remote_batches_per_session: int,
) -> tuple[np.ndarray, float]:
    """Run the deployed left-padded direct-margin implementation."""
    scores = np.full(rows, np.nan, dtype=np.float64)
    group_size = remote_batches_per_session or len(batches)
    started = time.perf_counter()
    for group_start in range(0, len(batches), group_size):
        group = batches[group_start : group_start + group_size]
        pieces = []
        print(
            "padded traces "
            f"{group_start + 1}-{group_start + len(group)}/{len(batches)} "
            f"shapes={[(len(p), int(e['input_ids'].shape[1])) for p, e in group]}",
            flush=True,
        )
        with model.session(remote=True):
            for _, encoded in group:
                with model.trace(
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "logits_to_keep": 1,
                    }
                ):
                    logits = model.output.logits[:, -1, label_ids].float()
                    pieces.append(
                        torch.softmax(logits, dim=-1)[:, 1].detach().cpu()
                    )
            saved = torch.cat(pieces, dim=0).save()
        values = np.asarray(saved.float().tolist(), dtype=np.float64)
        cursor = 0
        for positions, _ in group:
            count = len(positions)
            scores[positions] = values[cursor : cursor + count]
            cursor += count
    return scores, time.perf_counter() - started


def score_packed(
    model: Any,
    label_ids: list[int],
    batches: list[tuple[list[int], dict[str, Any]]],
    *,
    rows: int,
    remote_batches_per_session: int,
) -> tuple[np.ndarray, float]:
    """Run padding-free Qwen3.5 with explicit attention/GDN/conv boundaries."""
    scores = np.full(rows, np.nan, dtype=np.float64)
    group_size = remote_batches_per_session or len(batches)
    started = time.perf_counter()
    for group_start in range(0, len(batches), group_size):
        group = batches[group_start : group_start + group_size]
        pieces = []
        print(
            "packed traces "
            f"{group_start + 1}-{group_start + len(group)}/{len(batches)} "
            f"shapes={[(len(p), int(e['input_ids'].shape[1])) for p, e in group]}",
            flush=True,
        )
        with model.session(remote=True):
            for _, encoded in group:
                with model.trace(
                    {
                        "input_ids": encoded["input_ids"],
                        "position_ids": encoded["position_ids"],
                        "seq_idx": encoded["seq_idx"],
                        "cu_seq_lens_q": encoded["cu_seq_lens_q"],
                        "cu_seq_lens_k": encoded["cu_seq_lens_k"],
                        "max_length_q": encoded["max_length_q"],
                        "max_length_k": encoded["max_length_k"],
                        "logits_to_keep": encoded["logits_to_keep"],
                    }
                ):
                    logits = model.output.logits[:, :, label_ids].float()
                    pieces.append(
                        torch.softmax(logits, dim=-1)[0, :, 1]
                        .detach()
                        .cpu()
                    )
            saved = torch.cat(pieces, dim=0).save()
        values = np.asarray(saved.float().tolist(), dtype=np.float64)
        cursor = 0
        for positions, _ in group:
            count = len(positions)
            scores[positions] = values[cursor : cursor + count]
            cursor += count
    return scores, time.perf_counter() - started


def workload_report(
    padded_batches: list[tuple[list[int], Any]],
    packed_batches: list[tuple[list[int], dict[str, Any]]],
    token_ids: list[list[int]],
) -> dict[str, Any]:
    input_tokens = int(sum(len(ids) for ids in token_ids))
    padded_tokens = int(sum(
        len(positions) * int(encoded["input_ids"].shape[1])
        for positions, encoded in padded_batches
    ))
    packed_tokens = int(sum(
        int(encoded["input_ids"].shape[1])
        for _, encoded in packed_batches
    ))
    return {
        "rows": len(token_ids),
        "input_tokens": input_tokens,
        "padded_tokens": padded_tokens,
        "padding_tokens": padded_tokens - input_tokens,
        "padding_fraction": 1.0 - input_tokens / padded_tokens,
        "packed_tokens": packed_tokens,
        "padded_traces": len(padded_batches),
        "packed_traces": len(packed_batches),
        "padded_shapes": [
            [len(positions), int(encoded["input_ids"].shape[1])]
            for positions, encoded in padded_batches
        ],
        "packed_shapes": [
            [len(positions), int(encoded["input_ids"].shape[1])]
            for positions, encoded in packed_batches
        ],
    }


def main() -> None:
    args = parse_args()
    load_credentials()
    limit = None if args.limit == 0 else args.limit
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.splits_dir.resolve(), limit)
    spec = next(item for item in ADAPTERS if item.name == "qwen27_soft")

    from nnsight import LanguageModel

    model = LanguageModel("Qwen/Qwen3.5-9B", peft=spec.repo_id)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = binary_token_ids(tokenizer)

    prompts, renderer_parity = render_prompts(
        records,
        tokenizer,
        prompt_templates(),
        ["binary"],
    )
    prompt_values = prompts["binary"]
    token_ids = tokenize_prompts(tokenizer, prompt_values)
    padded_batches, _ = encode_batches(tokenizer, prompt_values)
    packed_batches = encode_packed_batches(
        token_ids,
        token_budget=args.packed_token_budget,
        max_sequences=args.max_sequences_per_pack,
    )
    workload = workload_report(
        padded_batches,
        packed_batches,
        token_ids,
    )
    print(json.dumps(workload, indent=2), flush=True)

    padded_scores, padded_seconds = score_padded(
        model,
        label_ids,
        padded_batches,
        rows=len(records),
        remote_batches_per_session=args.remote_batches_per_session,
    )
    packed_scores, packed_seconds = score_packed(
        model,
        label_ids,
        packed_batches,
        rows=len(records),
        remote_batches_per_session=args.remote_batches_per_session,
    )
    if np.isnan(padded_scores).any() or np.isnan(packed_scores).any():
        raise RuntimeError("one or both scoring paths left missing values")

    comparison = paired_report(padded_scores, packed_scores)
    comparison["close_at_1e_5"] = int(
        np.isclose(
            padded_scores,
            packed_scores,
            atol=1e-5,
            rtol=0.0,
        ).sum()
    )
    report = {
        "model_id": "Qwen/Qwen3.5-9B",
        "adapter": spec.repo_id,
        "rows": len(records),
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "packed_token_budget": args.packed_token_budget,
        "max_sequences_per_pack": args.max_sequences_per_pack,
        "renderer_parity": renderer_parity,
        "workload": workload,
        "timing": {
            "padded_seconds": padded_seconds,
            "packed_seconds": packed_seconds,
            "speedup": padded_seconds / packed_seconds,
        },
        "score_parity": comparison,
        "padded_metrics": condition_metrics(records, padded_scores),
        "packed_metrics": condition_metrics(records, packed_scores),
        "pooled_padded": binary_metrics(
            records["label"].to_numpy(),
            padded_scores,
        ),
        "pooled_packed": binary_metrics(
            records["label"].to_numpy(),
            packed_scores,
        ),
    }
    predictions = records.drop(columns="messages").copy()
    predictions["padded_score"] = padded_scores
    predictions["packed_score"] = packed_scores
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
