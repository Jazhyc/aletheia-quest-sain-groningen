#!/usr/bin/env python3
"""Build the direct-label-logit Phoenix Wright 4.0 submission notebook."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "phoenix_wright_v4_0.ipynb"


def main() -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            """# Phoenix Wright v4.0

AUROC-first black-box deception judge using the frozen Phoenix v3 rank-16
Qwen3.5-9B adapter. The notebook skips reasoning generation and reads the
normalized next-token margin between the exact labels `0` and `1` immediately
after `Prediction:`. The continuous margin is the submission score; a separately
frozen threshold supplies the required binary decision."""
        ),
        nbformat.v4.new_code_cell(
            """import contextlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "submission")
sys.path.insert(0, "experiments/heterogeneous_adapter_ensemble")
from phoenix_wright_v4 import (
    MEMBER_NAMES,
    binary_token_ids,
    build_direct_prompt,
)
import util

DATASET_NAME = os.environ["DATASET_NAME"]
LIMIT = int(os.environ["ALETHEIA_LIMIT"]) if os.environ.get("ALETHEIA_LIMIT") else None

JUDGE_MODEL = "Qwen/Qwen3.5-9B"
ADAPTER_REPO = os.environ.get(
    "PHOENIX_DECEPTION_ADAPTER_REPO",
    "Jazhyc/aletheias-phoenix-wright-v2-adapter",
)
BUNDLED_ADAPTER = "experiments/heterogeneous_adapter_ensemble/phoenix_wright_v3_adapters/deception"

MAX_PROMPT_TOKENS = int(os.environ.get("PHOENIX_MAX_PROMPT_TOKENS", "2048"))
BATCH_SIZE = int(os.environ.get("PHOENIX_BATCH_SIZE", "48"))
MEDIUM_BATCH_SIZE = int(os.environ.get("PHOENIX_MEDIUM_BATCH_SIZE", "32"))
MEDIUM_PROMPT_THRESHOLD = int(os.environ.get("PHOENIX_MEDIUM_PROMPT_THRESHOLD", "600"))
LONG_BATCH_SIZE = int(os.environ.get("PHOENIX_LONG_BATCH_SIZE", "16"))
LONG_PROMPT_THRESHOLD = int(os.environ.get("PHOENIX_LONG_PROMPT_THRESHOLD", "900"))
REMOTE_BATCHES_PER_SESSION = int(os.environ.get("PHOENIX_REMOTE_BATCHES_PER_SESSION", "0"))
NNSIGHT_LOCAL = os.environ.get("PHOENIX_NNSIGHT_LOCAL", "0").lower() in {"1", "true", "yes"}
SUBMISSION_THRESHOLD = float(os.environ.get("PHOENIX_THRESHOLD", "0.15"))

print(f"dataset={DATASET_NAME} limit={LIMIT}")
print(f"method=phoenix_wright_v4.0 direct_binary_margin members={MEMBER_NAMES}")
print(f"adapter_repo={ADAPTER_REPO}")
print(
    f"batch={BATCH_SIZE}/{MEDIUM_BATCH_SIZE}/{LONG_BATCH_SIZE} "
    f"sessions={REMOTE_BATCHES_PER_SESSION} max_prompt_tokens={MAX_PROMPT_TOKENS} "
    f"threshold={SUBMISSION_THRESHOLD} nnsight_local={NNSIGHT_LOCAL}"
)"""
        ),
        nbformat.v4.new_code_cell(
            """try:
    from nnsight import LanguageModel
except ImportError:
    from nnsight.modeling.language import LanguageModel

adapter_source = BUNDLED_ADAPTER if NNSIGHT_LOCAL else ADAPTER_REPO
model_kwargs = {"device_map": "auto", "dispatch": True, "dtype": "bfloat16"} if NNSIGHT_LOCAL else {}
model = LanguageModel(JUDGE_MODEL, peft=adapter_source, **model_kwargs)
tokenizer = model.tokenizer
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

BINARY_TOKEN_IDS = list(binary_token_ids(tokenizer))
print(
    f"adapter={adapter_source} binary_token_ids={BINARY_TOKEN_IDS} "
    f"pad_token_id={tokenizer.pad_token_id}"
)"""
        ),
        nbformat.v4.new_code_cell(
            """def make_position_batches(prompt_lengths):
    order = np.argsort(prompt_lengths)
    batches = []
    cursor = 0
    while cursor < len(order):
        cap = BATCH_SIZE
        candidate = order[cursor:min(cursor + cap, len(order))]
        longest = max(prompt_lengths[position] for position in candidate)
        if longest > MEDIUM_PROMPT_THRESHOLD:
            cap = min(cap, MEDIUM_BATCH_SIZE)
            candidate = order[cursor:min(cursor + cap, len(order))]
            longest = max(prompt_lengths[position] for position in candidate)
        if longest > LONG_PROMPT_THRESHOLD:
            cap = min(cap, LONG_BATCH_SIZE)
            candidate = order[cursor:min(cursor + cap, len(order))]
        batches.append(candidate.tolist())
        cursor += len(candidate)
    return batches


def score_prompts(prompts):
    if not prompts:
        return np.array([], dtype=float)
    prompt_lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    ]
    position_batches = make_position_batches(prompt_lengths)
    encoded_batches = []
    for positions in position_batches:
        encoded = tokenizer(
            [prompts[position] for position in positions],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        )
        encoded_batches.append((encoded, positions, encoded["input_ids"].shape[1]))

    batches_per_session = (
        len(encoded_batches)
        if NNSIGHT_LOCAL or REMOTE_BATCHES_PER_SESSION <= 0
        else REMOTE_BATCHES_PER_SESSION
    )
    saved_groups = []
    for group_start in range(0, len(encoded_batches), batches_per_session):
        group_stop = min(group_start + batches_per_session, len(encoded_batches))
        score_pieces = []
        session = contextlib.nullcontext() if NNSIGHT_LOCAL else model.session(remote=True)
        shapes = [
            (len(positions), prompt_tokens)
            for _, positions, prompt_tokens in encoded_batches[group_start:group_stop]
        ]
        print(
            f"direct-margin: batches {group_start + 1}-{group_stop}/"
            f"{len(encoded_batches)} shapes={shapes}",
            flush=True,
        )
        with session:
            for encoded, _, _ in encoded_batches[group_start:group_stop]:
                with model.trace({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "logits_to_keep": 1,
                }):
                    label_logits = model.output.logits[:, -1, BINARY_TOKEN_IDS].float()
                    piece = torch.softmax(label_logits, dim=-1)[:, 1].detach().cpu()
                    score_pieces.append(piece)
            group_scores = torch.cat(score_pieces, dim=0).save()
        saved_groups.append(group_scores)

    sorted_scores = torch.cat(saved_groups, dim=0).float().numpy()
    scores = np.empty(len(prompts), dtype=float)
    cursor = 0
    for _, positions, _ in encoded_batches:
        count = len(positions)
        scores[positions] = sorted_scores[cursor:cursor + count]
        cursor += count
    return np.clip(scores, 0.0, 1.0)


def score_messages(messages):
    prompts = [build_direct_prompt(value, tokenizer) for value in messages]
    scores = score_prompts(prompts)
    print(
        f"direct-margin: active={len(messages)} "
        f"unique_scores={len(np.unique(scores))}",
        flush=True,
    )
    return scores"""
        ),
        nbformat.v4.new_code_cell(
            """examples = util.load_examples(DATASET_NAME)
n = len(examples) if LIMIT is None else min(LIMIT, len(examples))
messages = list(examples["messages"][:n])
indices = list(examples["index"][:n])
print(f"loaded={len(examples)} scoring={n}")

scores = score_messages(messages)
print(
    f"score min={scores.min():.6f} mean={scores.mean():.6f} "
    f"max={scores.max():.6f} positives={int((scores >= SUBMISSION_THRESHOLD).sum())}"
)
util.write_submission(indices, scores, threshold=SUBMISSION_THRESHOLD)"""
        ),
    ]
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
