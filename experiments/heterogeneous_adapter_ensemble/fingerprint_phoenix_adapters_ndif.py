#!/usr/bin/env python3
"""Fingerprint hosted Phoenix adapters through the competition NDIF path."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.phoenix_wright import (  # noqa: E402
    binary_token_ids,
    build_main_direct_prompt,
)
from legacy_submissions.phoenix_wright_v3 import (  # noqa: E402
    build_member_prompt as build_legacy_member_prompt,
)


DEFAULT_DATASET = "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None"
DEFAULT_NDIF_HOST = "https://aletheias.api.ndif.us"
DEFAULT_OUTPUT = (
    ROOT
    / "results/blackbox/phoenix_adapter_ndif_fingerprint_v1/result.json"
)
ADAPTERS = {
    "base_qwen": None,
    "phoenix_4_0_gptoss_pi": "Jazhyc/aletheias-phoenix-wright-v2-adapter",
    "phoenix_5_1_gptoss_blind": (
        "Jazhyc/aletheias-phoenix-blind-reasoning-r16"
    ),
    "phoenix_5_2_luna_pi": (
        "Jazhyc/aletheias-phoenix-luna-medium-reasoning-r16"
    ),
}


def load_credentials() -> None:
    """Load local credentials without printing their values."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    os.environ.setdefault("NDIF_HOST", DEFAULT_NDIF_HOST)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is missing")


def hosted_metadata(repo_id: str | None) -> dict[str, Any]:
    """Return the immutable hosted revision and adapter-weight digest."""
    if repo_id is None:
        return {
            "repo_id": None,
            "revision": None,
            "weight_sha256": None,
            "weight_bytes": None,
        }
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    weights = next(
        sibling
        for sibling in info.siblings
        if sibling.rfilename == "adapter_model.safetensors"
    )
    lfs = getattr(weights, "lfs", None)
    return {
        "repo_id": repo_id,
        "revision": info.sha,
        "weight_sha256": lfs.sha256 if lfs is not None else None,
        "weight_bytes": weights.size,
    }


def score_adapter(
    repo_id: str | None,
    messages: list[Any],
    *,
    batch_size: int,
    max_prompt_tokens: int,
    renderer: str,
) -> dict[str, Any]:
    """Read direct binary margins for one hosted adapter."""
    from nnsight import LanguageModel

    model = (
        LanguageModel("Qwen/Qwen3.5-9B")
        if repo_id is None
        else LanguageModel("Qwen/Qwen3.5-9B", peft=repo_id)
    )
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = list(binary_token_ids(tokenizer))
    if renderer == "current_v5":
        prompts = [
            build_main_direct_prompt(message_row, tokenizer)
            for message_row in messages
        ]
    elif renderer == "legacy_v4":
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": build_legacy_member_prompt(row)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "Prediction:"
            for row in messages
        ]
    else:
        raise ValueError(f"unknown renderer: {renderer}")
    batches = [
        tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        for start in range(0, len(prompts), batch_size)
    ]

    started = time.perf_counter()
    with model.session(remote=True):
        pieces = []
        for batch in batches:
            with model.trace(
                {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "logits_to_keep": 1,
                }
            ):
                logits = model.output.logits[:, -1, label_ids].float()
                pieces.append(
                    torch.softmax(logits, dim=-1)[:, 1].detach().cpu()
                )
        saved_scores = torch.cat(pieces).save()
    elapsed = time.perf_counter() - started
    scores = np.asarray(saved_scores.float().tolist(), dtype=np.float64)
    result = {
        "prompt_sha256": hashlib.sha256(
            "\0".join(prompts).encode()
        ).hexdigest(),
        "score_sha256_float32": hashlib.sha256(
            scores.astype("<f4").tobytes()
        ).hexdigest(),
        "scores": scores.tolist(),
        "unique_scores": int(np.unique(scores).size),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "elapsed_seconds": elapsed,
    }
    del model
    gc.collect()
    return result


def compare_scores(
    left_name: str,
    left: list[float],
    right_name: str,
    right: list[float],
) -> dict[str, Any]:
    """Summarize whether two remote score fingerprints differ."""
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError(
            f"score shape mismatch: {left_values.shape} != {right_values.shape}"
        )
    return {
        "left": left_name,
        "right": right_name,
        "rows": int(left_values.size),
        "exact_equal_rows": int(np.sum(left_values == right_values)),
        "close_rows_at_1e_7": int(
            np.sum(np.isclose(left_values, right_values, atol=1e-7, rtol=0))
        ),
        "pearson": float(np.corrcoef(left_values, right_values)[0, 1]),
        "mean_absolute_difference": float(
            np.mean(np.abs(left_values - right_values))
        ),
        "max_absolute_difference": float(
            np.max(np.abs(left_values - right_values))
        ),
        "same_full_order": bool(
            np.array_equal(
                np.argsort(left_values, kind="stable"),
                np.argsort(right_values, kind="stable"),
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument(
        "--renderer",
        choices=("current_v5", "legacy_v4"),
        default="current_v5",
    )
    parser.add_argument(
        "--adapter-name",
        choices=tuple(ADAPTERS),
        help=(
            "score only this adapter; use separate processes per adapter to "
            "match leaderboard isolation"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 1 or args.batch_size < 1:
        raise ValueError("rows and batch size must be positive")
    load_credentials()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split="test").select(range(args.rows))
    messages = list(dataset["messages"])
    indices = list(dataset["index"])
    adapter_results: dict[str, Any] = {}
    selected_adapters = (
        {args.adapter_name: ADAPTERS[args.adapter_name]}
        if args.adapter_name is not None
        else ADAPTERS
    )
    for name, repo_id in selected_adapters.items():
        metadata = hosted_metadata(repo_id)
        print(
            f"starting {name} revision={metadata['revision']} "
            f"weight_sha256={metadata['weight_sha256']}",
            flush=True,
        )
        fingerprint = score_adapter(
            repo_id,
            messages,
            batch_size=args.batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            renderer=args.renderer,
        )
        adapter_results[name] = {**metadata, **fingerprint}
        print(
            f"finished {name} score_sha256="
            f"{fingerprint['score_sha256_float32']} "
            f"unique={fingerprint['unique_scores']} "
            f"elapsed={fingerprint['elapsed_seconds']:.3f}s",
            flush=True,
        )

    prompt_hashes = {
        result["prompt_sha256"] for result in adapter_results.values()
    }
    if len(prompt_hashes) != 1:
        raise RuntimeError(
            f"prompt hashes differ across adapters: {sorted(prompt_hashes)}"
        )
    names = list(adapter_results)
    comparisons = [
        compare_scores(
            left,
            adapter_results[left]["scores"],
            right,
            adapter_results[right]["scores"],
        )
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    ]
    result = {
        "dataset": args.dataset,
        "rows": args.rows,
        "batch_size": args.batch_size,
        "max_prompt_tokens": args.max_prompt_tokens,
        "renderer": args.renderer,
        "indices_sha256": hashlib.sha256(
            repr(indices).encode()
        ).hexdigest(),
        "shared_prompt_sha256": next(iter(prompt_hashes)),
        "adapters": adapter_results,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "adapters": {
        name: {
            key: value
            for key, value in values.items()
            if key != "scores"
        }
        for name, values in adapter_results.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
