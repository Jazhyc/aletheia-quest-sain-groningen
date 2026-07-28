#!/usr/bin/env python3
"""Compare safe short-prompt batch widths for Phoenix direct NDIF scoring."""

from __future__ import annotations

import argparse
import json
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
    binary_token_ids,
    condition_metrics,
    load_credentials,
    load_records,
    paired_report,
    prompt_templates,
    render_prompts,
)


DEFAULT_OUTPUT_DIR = (
    ROOT / "results/blackbox/phoenix_ndif_batch_width_v1"
)
SHORT_WIDTHS = (48, 52, 56)
MEDIUM_WIDTH = 32
LONG_WIDTH = 16
MEDIUM_THRESHOLD = 600
LONG_THRESHOLD = 900


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
        default=0,
        help="rows to score; 0 uses the complete 822-row validation split",
    )
    parser.add_argument(
        "--width",
        action="append",
        type=int,
        choices=SHORT_WIDTHS,
        help="repeat to select widths; defaults to 48, 52, and 56",
    )
    parser.add_argument(
        "--remote-batches-per-session",
        type=int,
        default=0,
        help="0 places all traces for one width in one remote session",
    )
    return parser.parse_args()


def position_batches(
    lengths: list[int],
    *,
    short_width: int,
) -> list[list[int]]:
    """Apply the submission tiers while varying only the short width."""
    if short_width <= 0:
        raise ValueError("short_width must be positive")
    order = np.argsort(lengths)
    batches: list[list[int]] = []
    cursor = 0
    while cursor < len(order):
        cap = short_width
        candidate = order[cursor : min(cursor + cap, len(order))]
        longest = max(lengths[int(position)] for position in candidate)
        if longest > MEDIUM_THRESHOLD:
            cap = min(cap, MEDIUM_WIDTH)
            candidate = order[cursor : min(cursor + cap, len(order))]
            longest = max(lengths[int(position)] for position in candidate)
        if longest > LONG_THRESHOLD:
            cap = min(cap, LONG_WIDTH)
            candidate = order[cursor : min(cursor + cap, len(order))]
        batches.append([int(position) for position in candidate])
        cursor += len(candidate)
    return batches


def tokenize_prompts(tokenizer: Any, prompts: list[str]) -> list[list[int]]:
    """Tokenize once with the submission's exact left-truncation contract."""
    return [
        tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        )
        for prompt in prompts
    ]


def encode_batches(
    tokenizer: Any,
    token_ids: list[list[int]],
    *,
    short_width: int,
) -> list[tuple[list[int], Any]]:
    lengths = [len(ids) for ids in token_ids]
    encoded = []
    for positions in position_batches(lengths, short_width=short_width):
        batch = tokenizer.pad(
            [{"input_ids": token_ids[position]} for position in positions],
            padding=True,
            return_tensors="pt",
        )
        encoded.append((positions, batch))
    return encoded


def workload_report(
    batches: list[tuple[list[int], Any]],
    token_ids: list[list[int]],
) -> dict[str, Any]:
    input_tokens = int(sum(len(ids) for ids in token_ids))
    padded_tokens = int(sum(
        len(positions) * int(encoded["input_ids"].shape[1])
        for positions, encoded in batches
    ))
    return {
        "rows": len(token_ids),
        "input_tokens": input_tokens,
        "padded_tokens": padded_tokens,
        "padding_tokens": padded_tokens - input_tokens,
        "padding_fraction": 1.0 - input_tokens / padded_tokens,
        "traces": len(batches),
        "max_padded_tokens_per_trace": int(max(
            len(positions) * int(encoded["input_ids"].shape[1])
            for positions, encoded in batches
        )),
        "shapes": [
            [len(positions), int(encoded["input_ids"].shape[1])]
            for positions, encoded in batches
        ],
    }


def score_batches(
    model: Any,
    label_ids: list[int],
    batches: list[tuple[list[int], Any]],
    *,
    rows: int,
    remote_batches_per_session: int,
    width: int | str,
) -> tuple[np.ndarray, float]:
    """Run one width in independent sessions so wall time is attributable."""
    scores = np.full(rows, np.nan, dtype=np.float64)
    group_size = remote_batches_per_session or len(batches)
    started = time.perf_counter()
    for group_start in range(0, len(batches), group_size):
        group = batches[group_start : group_start + group_size]
        pieces = []
        print(
            f"width={width} traces={group_start + 1}-"
            f"{group_start + len(group)}/{len(batches)} "
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
    if np.isnan(scores).any():
        raise RuntimeError(
            f"width {width} left {int(np.isnan(scores).sum())} missing scores"
        )
    return scores, time.perf_counter() - started


def write_partial_results(
    output_dir: Path,
    records: pd.DataFrame,
    report: dict[str, Any],
    scores_by_width: dict[int, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    predictions = records.drop(columns="messages").copy()
    for width, scores in scores_by_width.items():
        predictions[f"score_batch_{width}"] = scores
    predictions.to_csv(output_dir / "predictions.csv", index=False)


def main() -> None:
    args = parse_args()
    load_credentials()
    limit = None if args.limit == 0 else args.limit
    widths = args.width or list(SHORT_WIDTHS)
    output_dir = args.output_dir.resolve()
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
    token_ids = tokenize_prompts(tokenizer, prompts["binary"])

    batches_by_width = {
        width: encode_batches(
            tokenizer,
            token_ids,
            short_width=width,
        )
        for width in widths
    }
    report: dict[str, Any] = {
        "model_id": "Qwen/Qwen3.5-9B",
        "adapter": spec.repo_id,
        "rows": len(records),
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "widths": widths,
        "fixed_medium_width": MEDIUM_WIDTH,
        "fixed_long_width": LONG_WIDTH,
        "medium_threshold": MEDIUM_THRESHOLD,
        "long_threshold": LONG_THRESHOLD,
        "renderer_parity": renderer_parity,
        "conditions": {},
        "parity_vs_48": {},
    }
    scores_by_width: dict[int, np.ndarray] = {}
    for width in widths:
        batches = batches_by_width[width]
        workload = workload_report(batches, token_ids)
        print(json.dumps({"width": width, **workload}, indent=2), flush=True)
        scores, elapsed = score_batches(
            model,
            label_ids,
            batches,
            rows=len(records),
            remote_batches_per_session=args.remote_batches_per_session,
            width=width,
        )
        scores_by_width[width] = scores
        report["conditions"][str(width)] = {
            "workload": workload,
            "elapsed_seconds": elapsed,
            "metrics": condition_metrics(records, scores),
        }
        if 48 in scores_by_width and width != 48:
            report["parity_vs_48"][str(width)] = paired_report(
                scores_by_width[48],
                scores,
            )
        write_partial_results(
            output_dir,
            records,
            report,
            scores_by_width,
        )

    if 48 in scores_by_width:
        baseline_seconds = report["conditions"]["48"]["elapsed_seconds"]
        for width in widths:
            if width == 48:
                continue
            report["conditions"][str(width)]["speedup_vs_48"] = (
                baseline_seconds
                / report["conditions"][str(width)]["elapsed_seconds"]
            )
    write_partial_results(output_dir, records, report, scores_by_width)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
