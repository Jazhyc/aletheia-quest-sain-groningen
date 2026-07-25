#!/usr/bin/env python3
"""Cache the Phoenix v4 direct-label judge logits for every dev row.

This is the sonic v3 counterpart of ``run_judge_local.py``.  Where that script
reproduces the generative sonic judge (generate a reasoning summary, parse
``Prediction:<0|1>``, then teacher-force a soft confidence), this one runs the
Phoenix Wright 4.0 direct path: append ``Prediction:`` to the frozen judge
prompt and read the next-token logits of the exact labels ``0`` and ``1`` in a
single forward pass.  No generation, no parsing, no ties.

Both raw logits are cached rather than the softmax margin, because the sigmoid
saturates in float64 around +-37 and would collapse confident rows onto exactly
0.0 or 1.0 -- ties that cost AUROC directly.  The blend fit in
``fit_direct_blend.py`` consumes the raw difference.

The prompt builder is imported from the submission module itself, so the cached
scores are produced by the same code path the notebook will run.

FIDELITY: a local bf16 run is not bit-identical to the NDIF submission judge.
Use the cache for relative questions (blend weight, which datasets the judge
carries), not to predict an exact official score.

    python experiments/ensemble_gate_eval/run_direct_judge_local.py --limit 4 \
        --datasets aletheias-quest/dev-instructed-deception-Qwen3.5-27B-None
    python experiments/ensemble_gate_eval/run_direct_judge_local.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MANIFEST = REPO_ROOT / "dev_splits/manifest.csv"
DEFAULT_OUT = HERE / "direct_judge_cache.json"

sys.path.insert(0, str(REPO_ROOT / "experiments/heterogeneous_adapter_ensemble"))
from phoenix_wright_v4 import binary_token_ids, build_direct_prompt  # noqa: E402

JUDGE_MODEL = "Qwen/Qwen3.5-9B"
ADAPTER_REPO = "Jazhyc/aletheias-phoenix-wright-v2-adapter"
MAX_PROMPT_TOKENS = 2048


def load_judge(dtype: torch.dtype = torch.bfloat16) -> tuple[Any, Any]:
    """Load the frozen judge base model with its deception adapter attached."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, dtype=dtype, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_REPO)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def score_batch(model: Any, tokenizer: Any, prompts: list[str],
                label_ids: tuple[int, int], fp32_head: bool = False) -> np.ndarray:
    """Return the ``(rows, 2)`` label logits at the direct decision position.

    With ``fp32_head`` the two label logits are recomputed from the final
    hidden state in float32.  A bf16 logit near magnitude 20 has an ulp of
    0.125, which quantizes the ``logit_1 - logit_0`` margin onto roughly 35
    distinct values per dataset; those ties are a direct AUROC cost.
    """
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    ).to(model.device)
    if not fp32_head:
        output = model(**encoded, logits_to_keep=1, use_cache=False)
        return output.logits[:, -1, list(label_ids)].float().cpu().numpy()

    output = model(**encoded, logits_to_keep=1, use_cache=False,
                   output_hidden_states=True)
    hidden = output.hidden_states[-1][:, -1].float()
    head_weight = model.get_output_embeddings().weight[list(label_ids)].float()
    return (hidden @ head_weight.T).cpu().numpy()


def iter_rows(dataset_name: str, limit: int | None) -> Iterator[tuple[Any, Any]]:
    """Yield ``(index, messages)`` pairs, sampling evenly when limited.

    The dev datasets are ordered honest-first, so a prefix samples a single
    class; an evenly spaced sample keeps both classes present in smoke runs.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="test")
    if limit is None or limit >= len(dataset):
        positions: Any = range(len(dataset))
    else:
        spaced = np.linspace(0, len(dataset) - 1, limit).round().astype(int)
        positions = sorted({int(position) for position in spaced})
    for position in positions:
        example = dataset[position]
        yield example.get("index", position), example["messages"]


def manifest_datasets() -> list[str]:
    """Return every dataset name recorded in the frozen dev split manifest."""
    with open(MANIFEST) as handle:
        return sorted({row["dataset"] for row in csv.DictReader(handle)})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="dataset names; default = every manifest dataset")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="rows per dataset (smoke test)")
    parser.add_argument("--fp32-head", action="store_true",
                        help="recompute the two label logits in float32 to "
                             "avoid bf16 margin quantization (fewer ties)")
    args = parser.parse_args(argv)

    datasets = args.datasets or manifest_datasets()
    cache: dict[str, list[float]] = {}
    if args.out.exists():
        cache = json.loads(args.out.read_text())
        print(f"resuming from {args.out} ({len(cache)} rows cached)")

    model, tokenizer = load_judge()
    label_ids = binary_token_ids(tokenizer)
    print(f"judge = {JUDGE_MODEL} + {ADAPTER_REPO} (bf16)")
    print(f"label token ids = {label_ids}")

    for dataset in datasets:
        rows = list(iter_rows(dataset, args.limit))
        pending = [(index, messages) for index, messages in rows
                   if f"{dataset}|{index}" not in cache]
        short_name = dataset.split("/")[-1]
        if not pending:
            print(f"  {short_name}: all {len(rows)} rows cached, skip")
            continue

        prompts = {index: build_direct_prompt(messages, tokenizer)
                   for index, messages in pending}
        lengths = {index: len(tokenizer.encode(prompt, add_special_tokens=False))
                   for index, prompt in prompts.items()}
        ordered = sorted(prompts, key=lambda index: lengths[index])

        started = time.perf_counter()
        for start in range(0, len(ordered), args.batch_size):
            chunk = ordered[start:start + args.batch_size]
            logits = score_batch(model, tokenizer,
                                 [prompts[index] for index in chunk], label_ids,
                                 fp32_head=args.fp32_head)
            for index, row in zip(chunk, logits):
                cache[f"{dataset}|{index}"] = [float(row[0]), float(row[1])]
        args.out.write_text(json.dumps(cache) + "\n")
        elapsed = time.perf_counter() - started
        print(f"  {short_name}: {len(pending)} rows in {elapsed:.1f}s "
              f"({len(pending) / elapsed:.1f} rows/s)", flush=True)

    print(f"wrote {len(cache)} rows to {args.out}")


if __name__ == "__main__":
    main()
