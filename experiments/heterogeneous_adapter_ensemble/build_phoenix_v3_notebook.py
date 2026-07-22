#!/usr/bin/env python3
"""Build the single Phoenix Wright 3.0 submission notebook."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "submission/phoenix_wright_v3_0.ipynb"


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
            """# Phoenix Wright v3.0

Heterogeneous black-box ensemble over two rank-1 Qwen3.5-9B LoRA judges:
material deception and resolved deceptive intent. A frozen word 1--2 gram
logistic stack combines their binary votes with the visible conversation. The
intent member is skipped when assistant reasoning is unavailable or when its
two possible votes cannot cross the final binary threshold; skipped trace rows
use the intent logit midpoint for continuous scoring. Both adapter weights and
the n-gram model are bundled with the submission; NDIF loads matching public
Hugging Face adapter repositories."""
        ),
        nbformat.v4.new_code_cell(
            """import contextlib
import os
import sys

import joblib
import numpy as np
import torch

sys.path.insert(0, "submission")
from phoenix_wright_v3 import MEMBER_NAMES, build_member_prompt, has_reasoning, plan_intent_queries, reply_to_score, stack_scores
import util

DATASET_NAME = os.environ["DATASET_NAME"]
LIMIT = int(os.environ["ALETHEIA_LIMIT"]) if os.environ.get("ALETHEIA_LIMIT") else None

JUDGE_MODEL = "Qwen/Qwen3.5-9B"
ADAPTER_REPOS = {
    "deception": os.environ.get("PHOENIX_DECEPTION_ADAPTER_REPO", "Jazhyc/aletheias-phoenix-v3-deception-r1"),
    "resolved_intent": os.environ.get("PHOENIX_INTENT_ADAPTER_REPO", "Jazhyc/aletheias-phoenix-v3-resolved-intent-r1"),
}
BUNDLED_ADAPTERS = {
    name: f"submission/phoenix_wright_v3_adapters/{name}" for name in MEMBER_NAMES
}
STACK_PATH = "submission/phoenix_wright_v3_ngram.joblib"

MAX_PROMPT_TOKENS = int(os.environ.get("PHOENIX_MAX_PROMPT_TOKENS", "4096"))
MAX_NEW_TOKENS = int(os.environ.get("PHOENIX_MAX_NEW_TOKENS", "256"))
BATCH_SIZE = int(os.environ.get("PHOENIX_BATCH_SIZE", "32"))
MEDIUM_BATCH_SIZE = int(os.environ.get("PHOENIX_MEDIUM_BATCH_SIZE", "24"))
MEDIUM_PROMPT_THRESHOLD = int(os.environ.get("PHOENIX_MEDIUM_PROMPT_THRESHOLD", "1300"))
LONG_BATCH_SIZE = int(os.environ.get("PHOENIX_LONG_BATCH_SIZE", "16"))
LONG_PROMPT_THRESHOLD = int(os.environ.get("PHOENIX_LONG_PROMPT_THRESHOLD", "1800"))
# Zero means all batches for one adapter share a single NDIF request. Positive
# values retain bounded session groups for capacity/reliability diagnostics.
REMOTE_BATCHES_PER_SESSION = int(os.environ.get("PHOENIX_REMOTE_BATCHES_PER_SESSION", "0"))
NNSIGHT_LOCAL = os.environ.get("PHOENIX_NNSIGHT_LOCAL", "0").lower() in {"1", "true", "yes"}
SUBMISSION_THRESHOLD = float(os.environ.get("PHOENIX_THRESHOLD", "0.5"))
INTENT_GATE = os.environ.get("PHOENIX_INTENT_GATE", "1").lower() not in {"0", "false", "no"}

print(f"dataset={DATASET_NAME} limit={LIMIT}")
print(f"method=phoenix_wright_v3.0 members={MEMBER_NAMES}")
print(f"adapter_repos={ADAPTER_REPOS}")
print(
    f"batch={BATCH_SIZE}/{MEDIUM_BATCH_SIZE}/{LONG_BATCH_SIZE} "
    f"sessions={REMOTE_BATCHES_PER_SESSION} max_prompt_tokens={MAX_PROMPT_TOKENS} "
    f"max_new_tokens={MAX_NEW_TOKENS} nnsight_local={NNSIGHT_LOCAL} "
    f"intent_gate={INTENT_GATE}"
)"""
        ),
        nbformat.v4.new_code_cell(
            """try:
    from nnsight import LanguageModel
except ImportError:
    from nnsight.modeling.language import LanguageModel

initial_adapter = BUNDLED_ADAPTERS[MEMBER_NAMES[0]] if NNSIGHT_LOCAL else ADAPTER_REPOS[MEMBER_NAMES[0]]
model_kwargs = {"device_map": "auto", "dispatch": True, "dtype": "bfloat16"} if NNSIGHT_LOCAL else {}
model = LanguageModel(JUDGE_MODEL, peft=initial_adapter, **model_kwargs)
tokenizer = model.tokenizer
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

stack_artifact = joblib.load(STACK_PATH)
assert tuple(stack_artifact["member_names"]) == MEMBER_NAMES
print(
    f"pad_token_id={tokenizer.pad_token_id} eos_token_id={tokenizer.eos_token_id} "
    f"ngram_vocab={len(stack_artifact['vectorizer'].vocabulary_)}"
)"""
        ),
        nbformat.v4.new_code_cell(
            """def apply_judge_template(raw_prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def activate_adapter(member):
    source = BUNDLED_ADAPTERS[member] if NNSIGHT_LOCAL else ADAPTER_REPOS[member]
    if NNSIGHT_LOCAL:
        model._remoteable_set_env({"peft_repo_id": source})
    else:
        # Both LoRAs have the same rank/modules, so one meta-model can be
        # reused. NNsight sends the current repo in each session's NDIF env and
        # the updated server swaps PEFT weights without rebuilding the wrapper.
        model.__dict__["peft"] = source
    return source


def make_position_batches(prompt_lengths):
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


def score_prompts(prompts, member):
    if not prompts:
        return np.array([], dtype=float)
    source = activate_adapter(member)
    prompt_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
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

    generate_kwargs = {
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
    }
    batches_per_session = (
        len(encoded_batches)
        if NNSIGHT_LOCAL or REMOTE_BATCHES_PER_SESSION <= 0
        else REMOTE_BATCHES_PER_SESSION
    )
    generated_chunks = []
    for group_start in range(0, len(encoded_batches), batches_per_session):
        group_stop = min(group_start + batches_per_session, len(encoded_batches))
        generated_pieces = []
        session = contextlib.nullcontext() if NNSIGHT_LOCAL else model.session(remote=True)
        shapes = [
            (len(positions), prompt_tokens)
            for _, positions, prompt_tokens in encoded_batches[group_start:group_stop]
        ]
        print(
            f"{member}: adapter={source} batches {group_start + 1}-{group_stop}/"
            f"{len(encoded_batches)} shapes={shapes}",
            flush=True,
        )
        with session:
            for encoded, _, prompt_tokens in encoded_batches[group_start:group_stop]:
                with model.generate(
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                    },
                    **generate_kwargs,
                ):
                    piece = model.generator.output[:, prompt_tokens:].detach().cpu()
                    piece = torch.nn.functional.pad(
                        piece,
                        (0, MAX_NEW_TOKENS - piece.shape[1]),
                        value=tokenizer.pad_token_id,
                    )
                    generated_pieces.append(piece)
            generated_chunk = torch.cat(generated_pieces, dim=0).save()
        # Python container mutation must happen outside the trace context;
        # NNsight only returns explicitly saved proxy values across that boundary.
        generated_chunks.append(generated_chunk)

    generated_tokens = torch.cat(generated_chunks, dim=0)
    scores = np.zeros(len(prompts), dtype=float)
    cursor = 0
    for _, positions, _ in encoded_batches:
        tokens = generated_tokens[cursor:cursor + len(positions)]
        cursor += len(positions)
        replies = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        for position, reply in zip(positions, replies, strict=True):
            scores[position] = reply_to_score(reply)
    return scores


def score_member(messages, member, active_positions=None):
    scores = np.zeros(len(messages), dtype=float)
    if active_positions is None:
        if member == "resolved_intent":
            active_positions = [i for i, value in enumerate(messages) if has_reasoning(value)]
        else:
            active_positions = list(range(len(messages)))
    else:
        active_positions = list(active_positions)
    prompts = [
        apply_judge_template(build_member_prompt(messages[position], member))
        for position in active_positions
    ]
    if prompts:
        scores[active_positions] = score_prompts(prompts, member)
    print(
        f"{member}: active={len(active_positions)} routed_zero={len(messages) - len(active_positions)} "
        f"positive={int(scores.sum())}",
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

deception_scores = score_member(messages, "deception")
reasoning_mask = np.asarray([has_reasoning(value) for value in messages])
if INTENT_GATE:
    intent_query_mask, intent_features = plan_intent_queries(
        messages,
        deception_scores,
        stack_artifact,
        threshold=SUBMISSION_THRESHOLD,
    )
else:
    intent_query_mask = reasoning_mask
    intent_features = np.zeros(n, dtype=float)

intent_observed = score_member(
    messages,
    "resolved_intent",
    active_positions=np.flatnonzero(intent_query_mask),
)
intent_features[intent_query_mask] = intent_observed[intent_query_mask]
print(
    f"intent_gate: trace_rows={int(reasoning_mask.sum())} "
    f"queried={int(intent_query_mask.sum())} "
    f"skipped={int((reasoning_mask & ~intent_query_mask).sum())}",
    flush=True,
)

member_scores = np.column_stack([deception_scores, intent_features])
scores = stack_scores(messages, member_scores, stack_artifact)
print(
    f"stack score min={scores.min():.4f} mean={scores.mean():.4f} "
    f"max={scores.max():.4f} positives={int((scores >= SUBMISSION_THRESHOLD).sum())}"
)
util.write_submission(indices, scores, threshold=SUBMISSION_THRESHOLD)"""
        ),
    ]
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
