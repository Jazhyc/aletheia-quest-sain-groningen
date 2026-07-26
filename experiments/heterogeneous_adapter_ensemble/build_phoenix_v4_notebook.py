#!/usr/bin/env python3
"""Build the exclusive-specialist Phoenix Wright 4.3 notebook."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "submission/phoenix_wright_v4_3.ipynb"


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
            """# Phoenix Wright v4.3

AUROC-first white-box deception judge using the frozen Phoenix v3 rank-16
Qwen3.5-9B adapter. The notebook skips reasoning generation and reads the
normalized next-token margin between the exact labels `0` and `1`. The
continuous margin is the submission score; a separately frozen threshold
supplies the required binary decision.

Three routes remain mutually exclusive. Explicit harm-pressure knowledge
self-reports use the base-Qwen `knowledge_report_type` empty-summary specialist.
Executed-action reports use the frozen action-report prompt and empty-summary
margin on the ordinary Phoenix adapter. Remaining rows with a final assistant
reasoning trace combine Phoenix direct and rank-1 resolved-intent margins in
log-odds space. All other rows retain the exact Phoenix direct path. The
required binary column always uses the unblended route score."""
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
    blend_intent_route_scores,
    build_action_empty_prompt,
    build_direct_prompt,
    build_epistemic_empty_prompt,
    build_exclusive_route_flags,
    build_intent_empty_prompt,
    build_phoenix_route_prompt,
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
INTENT_ADAPTER_REPO = os.environ.get(
    "PHOENIX_INTENT_ADAPTER_REPO",
    "Jazhyc/aletheias-phoenix-v3-resolved-intent-r1",
)
BUNDLED_INTENT_ADAPTER = "submission/phoenix_wright_v4_adapters/resolved_intent"

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
print(f"method=phoenix_wright_v4.3 exclusive_action_margin members={MEMBER_NAMES}")
print(f"adapter_repo={ADAPTER_REPO}")
print(f"intent_adapter_repo={INTENT_ADAPTER_REPO}")
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
harm_routes, action_routes, intent_routes = build_exclusive_route_flags(messages)
print(
    f"loaded={len(examples)} scoring={n} "
    f"harm_specialist_routes={int(harm_routes.sum())} "
    f"action_specialist_routes={int(action_routes.sum())} "
    f"ordinary_intent_routes={int(intent_routes.sum())}"
)


def load_judge(*, kind):
    model_kwargs = (
        {"device_map": "auto", "dispatch": True, "dtype": "bfloat16"}
        if NNSIGHT_LOCAL
        else {}
    )
    if kind == "harm":
        active_model = LanguageModel(JUDGE_MODEL, **model_kwargs)
        source = "base"
    elif kind == "phoenix":
        adapter_source = BUNDLED_ADAPTER if NNSIGHT_LOCAL else ADAPTER_REPO
        active_model = LanguageModel(
            JUDGE_MODEL,
            peft=adapter_source,
            **model_kwargs,
        )
        source = adapter_source
    elif kind == "intent":
        adapter_source = (
            BUNDLED_INTENT_ADAPTER if NNSIGHT_LOCAL else INTENT_ADAPTER_REPO
        )
        active_model = LanguageModel(
            JUDGE_MODEL,
            peft=adapter_source,
            **model_kwargs,
        )
        source = adapter_source
    else:
        raise ValueError(f"unknown judge kind: {kind}")
    active_tokenizer = active_model.tokenizer
    active_tokenizer.padding_side = "left"
    active_tokenizer.truncation_side = "left"
    if active_tokenizer.pad_token_id is None:
        active_tokenizer.pad_token = active_tokenizer.eos_token
    label_ids = list(binary_token_ids(active_tokenizer))
    print(
        f"judge={kind} "
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


def score_message_subset(messages, positions, *, kind, action_flags=None):
    active_model, active_tokenizer, label_ids = load_judge(kind=kind)
    if kind == "phoenix":
        prompts = [
            build_phoenix_route_prompt(
                messages[position],
                active_tokenizer,
                action_report=action_flags[position],
            )
            for position in positions
        ]
    else:
        builder = {
            "harm": build_epistemic_empty_prompt,
            "intent": build_intent_empty_prompt,
        }[kind]
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
    action_count = (
        int(action_flags[positions].sum())
        if kind == "phoenix" and positions
        else 0
    )
    print(
        f"{kind}: active={len(positions)} action_prompts={action_count} "
        f"unique_scores={len(np.unique(subset_scores))}",
        flush=True,
    )
    return subset_scores


def score_messages(messages, harm_flags, action_flags, intent_flags):
    if np.any(harm_flags & action_flags):
        raise ValueError("harm and action routes must be disjoint")
    if np.any(harm_flags & intent_flags):
        raise ValueError("harm and intent routes must be disjoint")
    if np.any(action_flags & intent_flags):
        raise ValueError("action and intent routes must be disjoint")

    base_scores = np.empty(len(messages), dtype=float)
    phoenix_positions = np.flatnonzero(~harm_flags).tolist()
    if phoenix_positions:
        base_scores[phoenix_positions] = score_message_subset(
            messages,
            phoenix_positions,
            kind="phoenix",
            action_flags=action_flags,
        )
    harm_positions = np.flatnonzero(harm_flags).tolist()
    if harm_positions:
        base_scores[harm_positions] = score_message_subset(
            messages,
            harm_positions,
            kind="harm",
        )
    intent_positions = np.flatnonzero(intent_flags).tolist()
    intent_scores = (
        score_message_subset(messages, intent_positions, kind="intent")
        if intent_positions
        else np.array([], dtype=float)
    )
    scores = blend_intent_route_scores(base_scores, intent_scores, intent_flags)
    return scores, base_scores"""
        ),
        nbformat.v4.new_code_cell(
            """scores, binary_scores = score_messages(
    messages,
    harm_routes,
    action_routes,
    intent_routes,
)
deceptive = binary_scores >= SUBMISSION_THRESHOLD
print(
    f"score min={scores.min():.6f} mean={scores.mean():.6f} "
    f"max={scores.max():.6f} binary_positives={int(deceptive.sum())}"
)
util.write_submission(
    indices,
    scores,
    threshold=SUBMISSION_THRESHOLD,
    deceptive=deceptive,
)"""
        ),
    ]
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
