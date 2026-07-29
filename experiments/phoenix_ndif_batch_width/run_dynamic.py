#!/usr/bin/env python3
"""Evaluate dynamic padded-token batching against the frozen tier baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


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
from experiments.phoenix_ndif_batch_width.run import (
    DEFAULT_OUTPUT_DIR,
    score_batches,
    tokenize_prompts,
    workload_report,
)


PADDED_TOKEN_BUDGET = 28_800
ROW_CAPS = (48, 56)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=ROOT / "dev_splits",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "predictions.csv",
    )
    parser.add_argument(
        "--padded-token-budget",
        type=int,
        default=PADDED_TOKEN_BUDGET,
    )
    parser.add_argument(
        "--row-cap",
        action="append",
        type=int,
        choices=ROW_CAPS,
        help="repeat to select caps; defaults to 48 and 56",
    )
    parser.add_argument(
        "--remote-batches-per-session",
        type=int,
        default=0,
    )
    return parser.parse_args()


def dynamic_position_batches(
    lengths: list[int],
    *,
    row_cap: int,
    padded_token_budget: int,
) -> list[list[int]]:
    """Take the largest next batch satisfying row and padded-token caps."""
    if row_cap <= 0:
        raise ValueError("row_cap must be positive")
    if padded_token_budget <= 0:
        raise ValueError("padded_token_budget must be positive")
    if any(length <= 0 for length in lengths):
        raise ValueError("prompt lengths must be positive")

    order = np.argsort(lengths)
    batches: list[list[int]] = []
    cursor = 0
    while cursor < len(order):
        count = min(row_cap, len(order) - cursor)
        while count > 1:
            longest = lengths[int(order[cursor + count - 1])]
            if count * longest <= padded_token_budget:
                break
            count -= 1
        positions = [
            int(position)
            for position in order[cursor : cursor + count]
        ]
        batches.append(positions)
        cursor += count
    return batches


def encode_dynamic_batches(
    tokenizer: Any,
    token_ids: list[list[int]],
    *,
    row_cap: int,
    padded_token_budget: int,
) -> list[tuple[list[int], Any]]:
    lengths = [len(ids) for ids in token_ids]
    encoded = []
    for positions in dynamic_position_batches(
        lengths,
        row_cap=row_cap,
        padded_token_budget=padded_token_budget,
    ):
        batch = tokenizer.pad(
            [{"input_ids": token_ids[position]} for position in positions],
            padding=True,
            return_tensors="pt",
        )
        encoded.append((positions, batch))
    return encoded


def load_baseline(
    path: Path,
    records: pd.DataFrame,
) -> np.ndarray:
    baseline = pd.read_csv(path.resolve())
    required = {"dataset", "index", "score_batch_48"}
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(f"baseline predictions are missing columns: {missing}")
    expected_keys = list(zip(records["dataset"], records["index"], strict=True))
    found_keys = list(zip(baseline["dataset"], baseline["index"], strict=True))
    if found_keys != expected_keys:
        raise ValueError("baseline prediction keys do not match validation rows")
    return baseline["score_batch_48"].to_numpy(dtype=np.float64)


def write_results(
    output_dir: Path,
    records: pd.DataFrame,
    report: dict[str, Any],
    baseline_scores: np.ndarray,
    dynamic_scores: dict[int, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dynamic_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    predictions = records.drop(columns="messages").copy()
    predictions["score_batch_48"] = baseline_scores
    for row_cap, scores in dynamic_scores.items():
        predictions[f"score_dynamic_{row_cap}"] = scores
    predictions.to_csv(
        output_dir / "dynamic_predictions.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    load_credentials()
    row_caps = args.row_cap or list(ROW_CAPS)
    records = load_records(args.splits_dir.resolve())
    baseline_scores = load_baseline(args.baseline_predictions, records)
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

    report: dict[str, Any] = {
        "model_id": "Qwen/Qwen3.5-9B",
        "adapter": spec.repo_id,
        "rows": len(records),
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "padded_token_budget": args.padded_token_budget,
        "row_caps": row_caps,
        "renderer_parity": renderer_parity,
        "baseline_batch_48_metrics": condition_metrics(
            records,
            baseline_scores,
        ),
        "conditions": {},
    }
    dynamic_scores: dict[int, np.ndarray] = {}
    for row_cap in row_caps:
        batches = encode_dynamic_batches(
            tokenizer,
            token_ids,
            row_cap=row_cap,
            padded_token_budget=args.padded_token_budget,
        )
        workload = workload_report(batches, token_ids)
        print(
            json.dumps(
                {"condition": f"dynamic_{row_cap}", **workload},
                indent=2,
            ),
            flush=True,
        )
        scores, elapsed = score_batches(
            model,
            label_ids,
            batches,
            rows=len(records),
            remote_batches_per_session=args.remote_batches_per_session,
            width=f"dynamic_{row_cap}",
        )
        dynamic_scores[row_cap] = scores
        report["conditions"][str(row_cap)] = {
            "workload": workload,
            "elapsed_seconds": elapsed,
            "metrics": condition_metrics(records, scores),
            "parity_vs_batch_48": paired_report(
                baseline_scores,
                scores,
            ),
        }
        write_results(
            args.output_dir.resolve(),
            records,
            report,
            baseline_scores,
            dynamic_scores,
        )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
