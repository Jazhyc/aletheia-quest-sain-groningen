#!/usr/bin/env python3
"""Build the single deception-only Phoenix Wright 3.0 submission notebook."""

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

Black-box deception judge using one rank-1 Qwen3.5-9B LoRA. The adapter assesses
whether the visible final answer is materially false or misleading and emits a
binary prediction after its evidence summary. The exact adapter is bundled with
the submission and available from its matching public Hugging Face repository.
No lexical model or secondary intent judge is used."""
        ),
        nbformat.v4.new_code_cell(
            """import contextlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "submission")
from phoenix_wright_v3 import MEMBER_NAMES, build_member_prompt, reply_to_score
import util

DATASET_NAME = os.environ["DATASET_NAME"]
LIMIT = int(os.environ["ALETHEIA_LIMIT"]) if os.environ.get("ALETHEIA_LIMIT") else None

JUDGE_MODEL = "Qwen/Qwen3.5-9B"
ADAPTER_REPO = os.environ.get(
    "PHOENIX_DECEPTION_ADAPTER_REPO",
    "Jazhyc/aletheias-phoenix-v3-deception-r1",
)
BUNDLED_ADAPTER = "submission/phoenix_wright_v3_adapters/deception"

MAX_PROMPT_TOKENS = int(os.environ.get("PHOENIX_MAX_PROMPT_TOKENS", "2048"))
MAX_NEW_TOKENS = int(os.environ.get("PHOENIX_MAX_NEW_TOKENS", "256"))
BATCH_SIZE = int(os.environ.get("PHOENIX_BATCH_SIZE", "32"))
MEDIUM_BATCH_SIZE = int(os.environ.get("PHOENIX_MEDIUM_BATCH_SIZE", "24"))
MEDIUM_PROMPT_THRESHOLD = int(os.environ.get("PHOENIX_MEDIUM_PROMPT_THRESHOLD", "1300"))
LONG_BATCH_SIZE = int(os.environ.get("PHOENIX_LONG_BATCH_SIZE", "16"))
LONG_PROMPT_THRESHOLD = int(os.environ.get("PHOENIX_LONG_PROMPT_THRESHOLD", "1800"))
# Zero means all batches share one NDIF request. Positive values retain bounded
# session groups for capacity/reliability diagnostics.
REMOTE_BATCHES_PER_SESSION = int(os.environ.get("PHOENIX_REMOTE_BATCHES_PER_SESSION", "0"))
NNSIGHT_LOCAL = os.environ.get("PHOENIX_NNSIGHT_LOCAL", "0").lower() in {"1", "true", "yes"}
SUBMISSION_THRESHOLD = float(os.environ.get("PHOENIX_THRESHOLD", "0.5"))

print(f"dataset={DATASET_NAME} limit={LIMIT}")
print(f"method=phoenix_wright_v3.0 members={MEMBER_NAMES}")
print(f"adapter_repo={ADAPTER_REPO}")
print(
    f"batch={BATCH_SIZE}/{MEDIUM_BATCH_SIZE}/{LONG_BATCH_SIZE} "
    f"sessions={REMOTE_BATCHES_PER_SESSION} max_prompt_tokens={MAX_PROMPT_TOKENS} "
    f"max_new_tokens={MAX_NEW_TOKENS} nnsight_local={NNSIGHT_LOCAL}"
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

print(
    f"adapter={adapter_source} pad_token_id={tokenizer.pad_token_id} "
    f"eos_token_id={tokenizer.eos_token_id}"
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


def score_prompts(prompts):
    if not prompts:
        return np.array([], dtype=float)
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
            f"deception: batches {group_start + 1}-{group_stop}/"
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
        # NNsight returns only explicitly saved proxy values across the boundary.
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


def score_messages(messages):
    prompts = [apply_judge_template(build_member_prompt(value)) for value in messages]
    scores = score_prompts(prompts)
    print(
        f"deception: active={len(messages)} positive={int(scores.sum())}",
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
    f"score min={scores.min():.4f} mean={scores.mean():.4f} "
    f"max={scores.max():.4f} positives={int((scores >= SUBMISSION_THRESHOLD).sum())}"
)
util.write_submission(indices, scores, threshold=SUBMISSION_THRESHOLD)"""
        ),
    ]
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
