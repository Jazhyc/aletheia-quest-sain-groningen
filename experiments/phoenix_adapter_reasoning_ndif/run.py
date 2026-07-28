#!/usr/bin/env python3
"""Measure corrected post-reasoning AUROC for migrated Phoenix adapters."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
    AdapterSpec,
    MAX_PROMPT_TOKENS,
    MODEL_ID,
    binary_token_ids,
    condition_metrics,
    hosted_metadata,
    load_credentials,
    load_records,
    paired_report,
    position_batches,
    prompt_templates,
    training_member_prompt,
)
from submission.phoenix_wright import (
    has_binary_prediction,
    prefix_before_prediction,
)


ADAPTERS = (
    AdapterSpec(
        "gptoss_pi",
        "Jazhyc/aletheias-phoenix-wright-v2-1-adapter",
        "summary",
    ),
    AdapterSpec(
        "luna_pi",
        "Jazhyc/aletheias-phoenix-luna-medium-reasoning-r16",
        "summary",
    ),
)
MAX_NEW_TOKENS = 512
GENERATION_PADDED_TOKEN_BUDGET = 32_768
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/blackbox/phoenix_adapter_reasoning_ndif_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=ROOT / "dev_splits",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--adapter",
        action="append",
        choices=[spec.name for spec in ADAPTERS],
        help="repeat to score a subset; defaults to both adapters",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--generation-batches-per-session",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--score-batches-per-session",
        type=int,
        default=0,
    )
    return parser.parse_args()


def render_generation_prompts(
    frame: pd.DataFrame,
    tokenizer: Any,
    prompt_template: str,
) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": training_member_prompt(
                        messages,
                        prompt_template,
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in frame["messages"]
    ]


def generation_position_batches(lengths: list[int]) -> list[list[int]]:
    """Use Phoenix's generation tiers and padded input/output ceiling."""
    order = np.argsort(lengths)
    batches: list[list[int]] = []
    cursor = 0
    while cursor < len(order):
        cap = 32
        candidate = order[cursor : min(cursor + cap, len(order))]
        longest = max(lengths[int(position)] for position in candidate)
        if longest > 600:
            cap = min(cap, 24)
            candidate = order[cursor : min(cursor + cap, len(order))]
            longest = max(
                lengths[int(position)] for position in candidate
            )
        if longest > 900:
            cap = min(cap, 16)
            candidate = order[cursor : min(cursor + cap, len(order))]
            longest = max(
                lengths[int(position)] for position in candidate
            )
        cap = min(
            cap,
            max(
                1,
                GENERATION_PADDED_TOKEN_BUDGET
                // (longest + MAX_NEW_TOKENS),
            ),
        )
        candidate = order[cursor : min(cursor + cap, len(order))]
        batches.append([int(position) for position in candidate])
        cursor += len(candidate)
    return batches


def encode_generation_batches(
    tokenizer: Any,
    prompts: list[str],
) -> tuple[list[tuple[list[int], Any, int]], dict[str, Any]]:
    full_ids = [
        tokenizer.encode(prompt, add_special_tokens=False)
        for prompt in prompts
    ]
    input_ids = [ids[-MAX_PROMPT_TOKENS:] for ids in full_ids]
    lengths = [len(ids) for ids in input_ids]
    batches = []
    for positions in generation_position_batches(lengths):
        encoded = tokenizer.pad(
            [{"input_ids": input_ids[position]} for position in positions],
            padding=True,
            return_tensors="pt",
        )
        batches.append(
            (positions, encoded, int(encoded["input_ids"].shape[1]))
        )
    return batches, {
        "rows": len(prompts),
        "batches": len(batches),
        "min_tokens": min(len(ids) for ids in full_ids),
        "median_tokens": float(np.median([len(ids) for ids in full_ids])),
        "p95_tokens": float(np.quantile([len(ids) for ids in full_ids], 0.95)),
        "max_tokens": max(len(ids) for ids in full_ids),
        "truncated_rows": int(
            sum(len(ids) > MAX_PROMPT_TOKENS for ids in full_ids)
        ),
        "batch_shapes": [
            [len(positions), prompt_tokens]
            for positions, _, prompt_tokens in batches
        ],
        "max_padded_input_and_output_tokens": max(
            len(positions) * (prompt_tokens + MAX_NEW_TOKENS)
            for positions, _, prompt_tokens in batches
        ),
    }


def generate_reasoning(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    batches_per_session: int,
) -> tuple[list[str], dict[str, Any]]:
    batches, audit = encode_generation_batches(tokenizer, prompts)
    group_size = len(batches) if batches_per_session <= 0 else batches_per_session
    replies: list[str | None] = [None] * len(prompts)
    started = time.perf_counter()
    session_count = 0
    for group_start in range(0, len(batches), group_size):
        group = batches[group_start : group_start + group_size]
        session_count += 1
        print(
            f"generation session={session_count} batches="
            f"{group_start + 1}-{group_start + len(group)}/{len(batches)}",
            flush=True,
        )
        pieces = []
        with model.session(remote=True):
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
                    pieces.append(
                        torch.nn.functional.pad(
                            piece,
                            (0, MAX_NEW_TOKENS - piece.shape[1]),
                            value=tokenizer.pad_token_id,
                        )
                    )
            saved = torch.cat(pieces, dim=0).save()
        values = saved.long()
        cursor = 0
        for positions, _, _ in group:
            decoded = tokenizer.batch_decode(
                values[cursor : cursor + len(positions)],
                skip_special_tokens=True,
            )
            cursor += len(positions)
            for position, reply in zip(positions, decoded, strict=True):
                replies[position] = reply

    completed = [str(reply) for reply in replies]
    reply_lengths = [
        len(tokenizer.encode(reply, add_special_tokens=False))
        for reply in completed
    ]
    audit.update(
        {
            "seconds": time.perf_counter() - started,
            "sessions": session_count,
            "fallback_boundaries": int(
                sum(not has_binary_prediction(reply) for reply in completed)
            ),
            "reply_tokens": {
                "min": min(reply_lengths),
                "median": float(np.median(reply_lengths)),
                "p95": float(np.quantile(reply_lengths, 0.95)),
                "max": max(reply_lengths),
            },
        }
    )
    return completed, audit


def encode_score_batches(
    tokenizer: Any,
    prompt_sets: dict[str, list[str]],
) -> tuple[list[tuple[str, list[int], Any]], dict[str, Any]]:
    flat_batches = []
    audits = {}
    for name, prompts in prompt_sets.items():
        full_ids = [
            tokenizer.encode(prompt, add_special_tokens=False)
            for prompt in prompts
        ]
        lengths = [len(ids) for ids in full_ids]
        batches = []
        for positions in position_batches(lengths):
            encoded = tokenizer(
                [prompts[position] for position in positions],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_PROMPT_TOKENS,
            )
            batches.append((positions, encoded))
            flat_batches.append((name, positions, encoded))
        audits[name] = {
            "rows": len(prompts),
            "batches": len(batches),
            "min_tokens": min(lengths),
            "median_tokens": float(np.median(lengths)),
            "p95_tokens": float(np.quantile(lengths, 0.95)),
            "max_tokens": max(lengths),
            "truncated_rows": int(
                sum(length > MAX_PROMPT_TOKENS for length in lengths)
            ),
            "batch_shapes": [
                [len(positions), int(encoded["input_ids"].shape[1])]
                for positions, encoded in batches
            ],
        }
    return flat_batches, audits


def score_prompt_sets(
    model: Any,
    tokenizer: Any,
    label_ids: list[int],
    prompt_sets: dict[str, list[str]],
    *,
    batches_per_session: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    batches, audits = encode_score_batches(tokenizer, prompt_sets)
    scores = {
        name: np.full(len(prompts), np.nan, dtype=np.float64)
        for name, prompts in prompt_sets.items()
    }
    group_size = len(batches) if batches_per_session <= 0 else batches_per_session
    started = time.perf_counter()
    sessions = 0
    for group_start in range(0, len(batches), group_size):
        group = batches[group_start : group_start + group_size]
        sessions += 1
        print(
            f"scoring session={sessions} batches="
            f"{group_start + 1}-{group_start + len(group)}/{len(batches)}",
            flush=True,
        )
        pieces = []
        with model.session(remote=True):
            for _, _, encoded in group:
                with model.trace(
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "logits_to_keep": 1,
                    }
                ):
                    logits = model.output.logits[:, -1, label_ids].float()
                    pieces.append(
                        torch.softmax(logits, dim=-1)[:, 1]
                        .detach()
                        .cpu()
                    )
            saved = torch.cat(pieces, dim=0).save()
        values = np.asarray(saved.float().tolist(), dtype=np.float64)
        cursor = 0
        for name, positions, _ in group:
            count = len(positions)
            scores[name][positions] = values[cursor : cursor + count]
            cursor += count
    for name, values in scores.items():
        if np.isnan(values).any():
            raise RuntimeError(
                f"{name}: {int(np.isnan(values).sum())} missing scores"
            )
    return scores, {
        "seconds": time.perf_counter() - started,
        "sessions": sessions,
        "prompts": audits,
    }


def main() -> None:
    args = parse_args()
    load_credentials()
    selected_names = args.adapter or [spec.name for spec in ADAPTERS]
    selected_specs = [
        spec for spec in ADAPTERS if spec.name in selected_names
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.splits_dir.resolve(), args.limit)
    template = prompt_templates()["summary"]

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "split": "validation",
        "rows": len(records),
        "datasets": int(records["dataset"].nunique()),
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_padded_token_budget":
            GENERATION_PADDED_TOKEN_BUDGET,
        "adapters": {},
    }

    from nnsight import LanguageModel

    for spec in selected_specs:
        metadata = hosted_metadata(spec.repo_id)
        print(
            f"starting adapter={spec.name} revision={metadata['revision']}",
            flush=True,
        )
        model = LanguageModel(MODEL_ID, peft=spec.repo_id)
        tokenizer = model.tokenizer
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        label_ids = binary_token_ids(tokenizer)
        generation_prompts = render_generation_prompts(
            records,
            tokenizer,
            template,
        )
        replies, generation_audit = generate_reasoning(
            model,
            tokenizer,
            generation_prompts,
            batches_per_session=args.generation_batches_per_session,
        )
        prompt_sets = {
            "direct": [
                prompt + "Prediction:" for prompt in generation_prompts
            ],
            "post_reasoning": [
                prompt + prefix_before_prediction(reply)
                for prompt, reply in zip(
                    generation_prompts,
                    replies,
                    strict=True,
                )
            ],
        }
        scores, score_audit = score_prompt_sets(
            model,
            tokenizer,
            label_ids,
            prompt_sets,
            batches_per_session=args.score_batches_per_session,
        )
        conditions = {
            name: condition_metrics(records, values)
            for name, values in scores.items()
        }
        report["adapters"][spec.name] = {
            "spec": asdict(spec),
            "hosted": metadata,
            "generation": generation_audit,
            "scoring": score_audit,
            "conditions": conditions,
            "post_reasoning_vs_direct": paired_report(
                scores["direct"],
                scores["post_reasoning"],
            ),
            "macro_auroc_delta": (
                conditions["post_reasoning"]["all"]["macro_auroc"]
                - conditions["direct"]["all"]["macro_auroc"]
            ),
        }
        predictions = records.drop(columns="messages").copy()
        predictions["direct_score"] = scores["direct"]
        predictions["post_reasoning_score"] = scores["post_reasoning"]
        predictions["reasoning_reply"] = replies
        predictions.to_json(
            output_dir / f"{spec.name}.jsonl",
            orient="records",
            lines=True,
        )
        (output_dir / "report.partial.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            f"finished adapter={spec.name} direct="
            f"{conditions['direct']['all']['macro_auroc']:.6f} "
            f"post={conditions['post_reasoning']['all']['macro_auroc']:.6f} "
            f"fallbacks={generation_audit['fallback_boundaries']}",
            flush=True,
        )

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    partial = output_dir / "report.partial.json"
    if partial.exists():
        partial.unlink()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
