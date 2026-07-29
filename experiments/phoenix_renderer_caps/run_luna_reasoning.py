#!/usr/bin/env python3
"""Compare base direct, Luna direct, and Luna post-reasoning cap sensitivity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_renderer_caps.run import (
    CONDITIONS,
    LIARS_BENCH_ID,
    MODEL_ID,
    SAMPLE_SEED,
    binary_token_ids,
    build_member_prompt,
    load_insider_trading_records,
    metric_report,
    prepare_condition,
    score_condition_ndif,
    select_records,
)
from submission.phoenix_wright import (
    has_binary_prediction,
    prefix_before_prediction,
)


LUNA_ADAPTER = "Jazhyc/aletheias-phoenix-luna-medium-reasoning-r16"
SELECTED_CONDITIONS = (CONDITIONS[0], CONDITIONS[1], CONDITIONS[3])
SOURCE_MODEL = "llama-v3.3-70b-instruct"
SAMPLE_SIZE = 400
MAX_NEW_TOKENS = 512
GENERATION_PADDED_TOKEN_BUDGET = 32_768
BOOTSTRAP_SAMPLES = 2_000
_PREDICTION_RE = re.compile(r"(?im)\bPrediction\s*:\s*<?([01])>?")
DEFAULT_BASE_PREDICTIONS = (
    ROOT
    / "results/blackbox/phoenix_renderer_caps_ndif_llama400_v1/predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/blackbox/phoenix_renderer_caps_luna_reasoning_llama400_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=DEFAULT_BASE_PREDICTIONS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--remote-batches-per-session", type=int, default=32)
    return parser.parse_args()


def build_generation_prompt(
    tokenizer: Any,
    messages: Any,
    condition: Any,
) -> str:
    member_prompt, _ = build_member_prompt(messages, condition)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": member_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generation_position_batches(
    prompt_lengths: list[int],
) -> list[list[int]]:
    """Use Luna's 32/24/16 tiers plus an input-and-output token ceiling."""
    order = np.argsort(prompt_lengths)
    batches = []
    cursor = 0
    while cursor < len(order):
        cap = 32
        candidate = order[cursor:min(cursor + cap, len(order))]
        longest = max(prompt_lengths[int(position)] for position in candidate)
        if longest > 600:
            cap = min(cap, 24)
            candidate = order[cursor:min(cursor + cap, len(order))]
            longest = max(
                prompt_lengths[int(position)] for position in candidate
            )
        if longest > 900:
            cap = min(cap, 16)
            candidate = order[cursor:min(cursor + cap, len(order))]
            longest = max(
                prompt_lengths[int(position)] for position in candidate
            )
        cap = min(
            cap,
            max(
                1,
                GENERATION_PADDED_TOKEN_BUDGET
                // (longest + MAX_NEW_TOKENS),
            ),
        )
        candidate = order[cursor:min(cursor + cap, len(order))]
        batches.append([int(position) for position in candidate])
        cursor += len(candidate)
    return batches


def generate_reasoning(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_prompt_tokens: int,
    remote_batches_per_session: int,
) -> tuple[list[str], float, dict[str, Any]]:
    import torch

    full_ids = [
        tokenizer.encode(prompt, add_special_tokens=False)
        for prompt in prompts
    ]
    input_ids = [ids[-max_prompt_tokens:] for ids in full_ids]
    lengths = [len(ids) for ids in input_ids]
    position_batches = generation_position_batches(lengths)
    encoded_batches = []
    for positions in position_batches:
        encoded = tokenizer.pad(
            [{"input_ids": input_ids[position]} for position in positions],
            padding=True,
            return_tensors="pt",
        )
        encoded_batches.append(
            (positions, encoded, int(encoded["input_ids"].shape[1]))
        )

    if remote_batches_per_session <= 0:
        remote_batches_per_session = len(encoded_batches)
    replies: list[str | None] = [None] * len(prompts)
    started = time.perf_counter()
    session_count = 0
    for group_start in range(
        0,
        len(encoded_batches),
        remote_batches_per_session,
    ):
        group = encoded_batches[
            group_start:group_start + remote_batches_per_session
        ]
        session_count += 1
        print(
            "Luna generation session "
            f"{session_count}: shapes="
            f"{[(len(p), n) for p, _, n in group]}",
            flush=True,
        )
        with model.session(remote=True):
            pieces = []
            for _, encoded, prompt_tokens in group:
                with model.generate(
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                    },
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                ):
                    piece = (
                        model.generator.output[:, prompt_tokens:]
                        .detach()
                        .cpu()
                    )
                    pieces.append(torch.nn.functional.pad(
                        piece,
                        (0, MAX_NEW_TOKENS - piece.shape[1]),
                        value=tokenizer.pad_token_id,
                    ))
            saved = torch.cat(pieces, dim=0).save()
        values = saved.long()
        cursor = 0
        for positions, _, _ in group:
            decoded = tokenizer.batch_decode(
                values[cursor:cursor + len(positions)],
                skip_special_tokens=True,
            )
            cursor += len(positions)
            for position, reply in zip(positions, decoded, strict=True):
                replies[position] = reply

    completed = [str(reply) for reply in replies]
    reply_token_lengths = [
        len(tokenizer.encode(reply, add_special_tokens=False))
        for reply in completed
    ]
    report = {
        "seconds": time.perf_counter() - started,
        "batches": len(encoded_batches),
        "sessions": session_count,
        "input_truncated_rows": int(sum(
            len(ids) > max_prompt_tokens for ids in full_ids
        )),
        "fallback_boundaries": int(sum(
            not has_binary_prediction(reply) for reply in completed
        )),
        "reply_tokens": {
            "median": float(np.median(reply_token_lengths)),
            "p95": float(np.quantile(reply_token_lengths, 0.95)),
            "max": int(max(reply_token_lengths)),
        },
        "max_padded_input_and_output_tokens": int(max(
            len(positions) * (prompt_tokens + MAX_NEW_TOKENS)
            for positions, _, prompt_tokens in encoded_batches
        )),
    }
    return completed, report["seconds"], report


def score_frame_from_prompts(
    prompts: list[str],
    tokenizer: Any,
    *,
    max_prompt_tokens: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    token_ids = [
        tokenizer.encode(prompt, add_special_tokens=False)
        for prompt in prompts
    ]
    frame = pd.DataFrame({
        "input_token_ids": [
            ids[-max_prompt_tokens:] for ids in token_ids
        ],
        "prompt_tokens_effective": [
            min(len(ids), max_prompt_tokens) for ids in token_ids
        ],
    })
    return frame, {
        "rows": len(frame),
        "truncated_rows": int(sum(
            len(ids) > max_prompt_tokens for ids in token_ids
        )),
        "tokens": {
            "median": float(np.median([len(ids) for ids in token_ids])),
            "p95": float(np.quantile([len(ids) for ids in token_ids], 0.95)),
            "max": int(max(len(ids) for ids in token_ids)),
        },
    }


def generated_predictions(replies: list[str]) -> np.ndarray:
    """Extract the last generated binary prediction from every reply."""
    predictions = []
    for reply in replies:
        matches = list(_PREDICTION_RE.finditer(reply))
        if not matches:
            raise ValueError("reply does not contain a binary Prediction label")
        predictions.append(int(matches[-1].group(1)))
    return np.asarray(predictions, dtype=int)


def paired_stratified_bootstrap_delta(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SAMPLE_SEED,
) -> dict[str, float | int]:
    """Estimate a paired AUROC delta interval with label-stratified resampling."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=int)
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for bootstrap_index in range(samples):
        positions = np.concatenate([
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ])
        deltas[bootstrap_index] = (
            roc_auc_score(labels[positions], candidate[positions])
            - roc_auc_score(labels[positions], reference[positions])
        )
    return {
        "delta": float(
            roc_auc_score(labels, candidate)
            - roc_auc_score(labels, reference)
        ),
        "ci_95_low": float(np.quantile(deltas, 0.025)),
        "ci_95_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def load_base_scores(
    path: Path,
    records: pd.DataFrame,
) -> dict[str, np.ndarray]:
    base = pd.read_csv(path)
    expected = records[["index", "label"]].reset_index(drop=True)
    actual = base[["index", "label"]].reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("base predictions do not match the frozen row order")
    return {
        condition.name: base[f"score_{condition.name}"].to_numpy(dtype=float)
        for condition in SELECTED_CONDITIONS
    }


def main() -> None:
    args = parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    os.environ.setdefault("NDIF_HOST", "https://aletheias.api.ndif.us")
    if not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is missing")

    from nnsight import LanguageModel

    model = LanguageModel(MODEL_ID, peft=LUNA_ADAPTER)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = binary_token_ids(tokenizer)

    population = load_insider_trading_records()
    records = select_records(
        population,
        source_model=SOURCE_MODEL,
        sample_size=SAMPLE_SIZE,
    )
    base_scores = load_base_scores(args.base_predictions, records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = records[
        ["dataset", "scenario", "index", "source_model", "label"]
    ].copy()
    generations = predictions.copy()
    report: dict[str, Any] = {
        "dataset": LIARS_BENCH_ID,
        "population_rows": int(len(population)),
        "selected_rows": int(len(records)),
        "source_model": SOURCE_MODEL,
        "sample_size": SAMPLE_SIZE,
        "sample_seed": SAMPLE_SEED,
        "judge_model": MODEL_ID,
        "luna_adapter": LUNA_ADAPTER,
        "max_new_tokens": MAX_NEW_TOKENS,
        "conditions": {},
    }

    for condition in SELECTED_CONDITIONS:
        print(f"condition={condition.name}", flush=True)
        prepared, direct_audit = prepare_condition(
            records,
            tokenizer,
            condition,
        )
        direct_scores, direct_seconds, direct_execution = score_condition_ndif(
            model,
            tokenizer,
            label_ids,
            prepared,
            remote_batches_per_session=args.remote_batches_per_session,
        )

        generation_prompts = [
            build_generation_prompt(tokenizer, messages, condition)
            for messages in records["messages"]
        ]
        replies, generation_seconds, generation_report = generate_reasoning(
            model,
            tokenizer,
            generation_prompts,
            max_prompt_tokens=condition.max_prompt_tokens,
            remote_batches_per_session=args.remote_batches_per_session,
        )
        post_prompts = [
            prompt + prefix_before_prediction(reply)
            for prompt, reply in zip(
                generation_prompts,
                replies,
                strict=True,
            )
        ]
        post_frame, post_audit = score_frame_from_prompts(
            post_prompts,
            tokenizer,
            max_prompt_tokens=condition.max_prompt_tokens,
        )
        post_scores, post_seconds, post_execution = score_condition_ndif(
            model,
            tokenizer,
            label_ids,
            post_frame,
            remote_batches_per_session=args.remote_batches_per_session,
        )

        base = base_scores[condition.name]
        labels = records["label"].to_numpy(dtype=int)
        generated_binary = generated_predictions(replies)
        predictions[f"base_direct_{condition.name}"] = base
        predictions[f"luna_direct_{condition.name}"] = direct_scores
        predictions[f"luna_reasoning_{condition.name}"] = post_scores
        generations[f"reply_{condition.name}"] = replies
        report["conditions"][condition.name] = {
            "base_direct": {
                "metrics": metric_report(records, base),
            },
            "luna_direct": {
                "audit": direct_audit,
                "seconds": direct_seconds,
                "execution": direct_execution,
                "metrics": metric_report(records, direct_scores),
            },
            "luna_reasoning": {
                "generation": generation_report,
                "generated_binary_metrics": metric_report(
                    records,
                    generated_binary,
                ),
                "post_prompt": post_audit,
                "score_seconds": post_seconds,
                "score_execution": post_execution,
                "total_seconds": generation_seconds + post_seconds,
                "metrics": metric_report(records, post_scores),
            },
            "paired_auroc_deltas": {
                "luna_direct_minus_base_direct":
                    paired_stratified_bootstrap_delta(
                        labels,
                        base,
                        direct_scores,
                    ),
                "luna_reasoning_minus_luna_direct":
                    paired_stratified_bootstrap_delta(
                        labels,
                        direct_scores,
                        post_scores,
                    ),
                "luna_reasoning_minus_base_direct":
                    paired_stratified_bootstrap_delta(
                        labels,
                        base,
                        post_scores,
                    ),
            },
        }
        print(
            json.dumps(report["conditions"][condition.name], indent=2),
            flush=True,
        )

    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    generations.to_csv(args.output_dir / "generations.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
