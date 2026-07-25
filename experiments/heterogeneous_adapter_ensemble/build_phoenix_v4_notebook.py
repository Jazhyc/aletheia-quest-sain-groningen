#!/usr/bin/env python3
"""Build the routed direct-label-logit Phoenix Wright 4.1 notebook."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "submission/phoenix_wright_v4_1.ipynb"


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
            """# Phoenix Wright v4.1

AUROC-first white-box deception judge using the frozen Phoenix v3 rank-16
Qwen3.5-9B adapter. The notebook skips reasoning generation and reads the
normalized next-token margin between the exact labels `0` and `1` immediately
after `Prediction:`. The continuous margin is the submission score; a separately
frozen threshold supplies the required binary decision.

An exact observable-contract router sends explicit knowledge self-reports to
the confirmed base-Qwen `knowledge_report_type` specialist. Those rows use the
empty-summary label margin; all other rows retain the Phoenix direct margin."""
        ),
        nbformat.v4.new_code_cell(
            """import contextlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "submission")
from phoenix_wright_v4 import (
    MEMBER_NAMES,
    binary_token_ids,
    build_direct_prompt,
    build_epistemic_empty_prompt,
    is_knowledge_report_request,
)
import util

DATASET_NAME = os.environ["DATASET_NAME"]
LIMIT = int(os.environ["ALETHEIA_LIMIT"]) if os.environ.get("ALETHEIA_LIMIT") else None

JUDGE_MODEL = "Qwen/Qwen3.5-9B"
ADAPTER_REPO = os.environ.get(
    "PHOENIX_DECEPTION_ADAPTER_REPO",
    "Jazhyc/aletheias-phoenix-wright-v2-adapter",
)
BUNDLED_ADAPTER = "submission/phoenix_wright_v3_adapters/deception"

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
print(f"method=phoenix_wright_v4.1 routed_direct_binary_margin members={MEMBER_NAMES}")
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

examples = util.load_examples(DATASET_NAME)
n = len(examples) if LIMIT is None else min(LIMIT, len(examples))
messages = list(examples["messages"][:n])
indices = list(examples["index"][:n])
knowledge_routes = np.asarray(
    [is_knowledge_report_request(value) for value in messages],
    dtype=bool,
)
print(
    f"loaded={len(examples)} scoring={n} "
    f"knowledge_routes={int(knowledge_routes.sum())}"
)


def load_judge(*, specialist):
    model_kwargs = (
        {"device_map": "auto", "dispatch": True, "dtype": "bfloat16"}
        if NNSIGHT_LOCAL
        else {}
    )
    if specialist:
        active_model = LanguageModel(JUDGE_MODEL, **model_kwargs)
        source = "base"
    else:
        adapter_source = BUNDLED_ADAPTER if NNSIGHT_LOCAL else ADAPTER_REPO
        active_model = LanguageModel(
            JUDGE_MODEL,
            peft=adapter_source,
            **model_kwargs,
        )
        source = adapter_source
    active_tokenizer = active_model.tokenizer
    active_tokenizer.padding_side = "left"
    active_tokenizer.truncation_side = "left"
    if active_tokenizer.pad_token_id is None:
        active_tokenizer.pad_token = active_tokenizer.eos_token
    label_ids = list(binary_token_ids(active_tokenizer))
    print(
        f"judge={'epistemic' if specialist else 'phoenix'} "
        f"source={source} binary_token_ids={label_ids} "
        f"pad_token_id={active_tokenizer.pad_token_id}"
    )
    return active_model, active_tokenizer, label_ids"""
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


def score_prompts(active_model, active_tokenizer, label_ids, prompts):
    if not prompts:
        return np.array([], dtype=float)
    prompt_lengths = [
        len(active_tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    ]
    position_batches = make_position_batches(prompt_lengths)
    encoded_batches = []
    for positions in position_batches:
        encoded = active_tokenizer(
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
        session = (
            contextlib.nullcontext()
            if NNSIGHT_LOCAL
            else active_model.session(remote=True)
        )
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
                with active_model.trace({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "logits_to_keep": 1,
                }):
                    label_logits = active_model.output.logits[:, -1, label_ids].float()
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


def score_message_subset(messages, positions, *, specialist):
    active_model, active_tokenizer, label_ids = load_judge(
        specialist=specialist,
    )
    builder = build_epistemic_empty_prompt if specialist else build_direct_prompt
    prompts = [
        builder(messages[position], active_tokenizer)
        for position in positions
    ]
    subset_scores = score_prompts(
        active_model,
        active_tokenizer,
        label_ids,
        prompts,
    )
    print(
        f"{'epistemic-empty' if specialist else 'phoenix-direct'}: "
        f"active={len(positions)} unique_scores={len(np.unique(subset_scores))}",
        flush=True,
    )
    return subset_scores


def score_messages(messages, route_flags):
    scores = np.empty(len(messages), dtype=float)
    for specialist, positions in (
        (False, np.flatnonzero(~route_flags).tolist()),
        (True, np.flatnonzero(route_flags).tolist()),
    ):
        if not positions:
            continue
        scores[positions] = score_message_subset(
            messages,
            positions,
            specialist=specialist,
        )
    return scores"""
        ),
        nbformat.v4.new_code_cell(
            """scores = score_messages(messages, knowledge_routes)
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
