#!/usr/bin/env python3
"""Full multi-route Phoenix v6.3 judge evaluation on all 21 dev datasets.

Loads one adapter at a time (memory-constrained), scores all datasets, blends.

Usage:
    python experiments/ensemble_gate_eval/run_v63_full_judge.py --limit 10  # smoke
    python experiments/ensemble_gate_eval/run_v63_full_judge.py              # full
"""

from __future__ import annotations

import argparse, csv, json, sys, time, gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "submission"))
sys.path.insert(0, str(REPO_ROOT))

MANIFEST = REPO_ROOT / "dev_splits/manifest.csv"
CACHE_OUT = HERE / "v63_full_judge_cache.json"

from phoenix_wright import (
    binary_token_ids, blend_intent_route_scores,
    build_action_empty_prompt, build_main_direct_prompt,
    build_epistemic_empty_prompt, build_exclusive_route_flags,
    build_intent_empty_prompt,
)

JUDGE_MODEL = "Qwen/Qwen3.5-9B"
MAIN_ADAPTER = "Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16-ep2"
INTENT_ADAPTER = "Jazhyc/aletheias-phoenix-v3-resolved-intent-r1"
BUNDLED_INTENT = REPO_ROOT / "submission/phoenix_wright_adapters/resolved_intent"
MAX_PROMPT_TOKENS = 2048


def manifest_datasets():
    with open(MANIFEST) as f:
        return sorted({row["dataset"] for row in csv.DictReader(f)})


def load_judge(adapter):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, tokenizer


def make_batches(prompt_lengths):
    order = np.argsort(prompt_lengths)
    batches, cursor = [], 0
    while cursor < len(order):
        cap = 16
        candidate = order[cursor:min(cursor + cap, len(order))]
        longest = max(prompt_lengths[p] for p in candidate)
        if longest > 600:
            cap = min(cap, 8)
            candidate = order[cursor:min(cursor + cap, len(order))]
            longest = max(prompt_lengths[p] for p in candidate)
        if longest > 900:
            cap = min(cap, 4)
            candidate = order[cursor:min(cursor + cap, len(order))]
        batches.append(candidate.tolist())
        cursor += len(candidate)
    return batches


@torch.no_grad()
def score_prompts(model, tokenizer, label_ids, prompts):
    if not prompts:
        return np.array([], dtype=float)
    plens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    batches = make_batches(plens)
    scores = np.empty(len(prompts), dtype=float)
    for positions in batches:
        bp = [prompts[p] for p in positions]
        enc = tokenizer(bp, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_PROMPT_TOKENS).to(model.device)
        out = model(**enc, logits_to_keep=1, use_cache=False)
        probs = torch.softmax(out.logits[:, -1, list(label_ids)].float(), dim=-1)[:, 1].cpu().numpy()
        for i, p in enumerate(positions):
            scores[p] = float(probs[i])
    return np.clip(scores, 0.0, 1.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=CACHE_OUT)
    args = parser.parse_args(argv)

    datasets = args.datasets or manifest_datasets()
    print(f"Datasets: {len(datasets)}, limit={args.limit}")

    # ---- Phase 1: load all data, build prompts for main route ----
    all_data = {}
    for ds_name in datasets:
        short = ds_name.split("/")[-1]
        dataset = load_dataset(ds_name, split="test")
        if args.limit:
            dataset = dataset.select(range(min(args.limit, len(dataset))))
        examples = list(dataset)
        messages = [ex["messages"] for ex in examples]
        labels = np.array([ex.get("deceptive", False) for ex in examples])
        indices = [ex.get("index", i) for i, ex in enumerate(examples)]
        harm_f, action_f, intent_f = build_exclusive_route_flags(messages)
        all_data[ds_name] = dict(
            short=short, messages=messages, labels=labels, indices=indices,
            harm_f=harm_f, action_f=action_f, intent_f=intent_f, n=len(messages))
        print(f"  {short}: {len(messages)} rows, "
              f"harm={int(harm_f.sum())} action={int(action_f.sum())} "
              f"intent={int(intent_f.sum())}")

    # ---- Phase 2: main route (load main adapter, score all datasets) ----
    print("\n=== Main route (v6.3 qwen397) ===")
    main_model, main_tok = load_judge(MAIN_ADAPTER)
    main_lids = binary_token_ids(main_tok)
    print(f"  label_ids={main_lids}")

    for ds_name, data in all_data.items():
        t0 = time.perf_counter()
        prompts = []
        for i, msgs in enumerate(data["messages"]):
            if data["harm_f"][i]:
                b = build_epistemic_empty_prompt
            elif data["action_f"][i]:
                b = build_action_empty_prompt
            else:
                b = build_main_direct_prompt
            prompts.append(b(msgs, main_tok))
        data["base_scores"] = score_prompts(main_model, main_tok, main_lids, prompts)
        print(f"  {data['short']}: {data['n']} rows in {time.perf_counter()-t0:.1f}s, "
              f"scores [{data['base_scores'].min():.3f}, {data['base_scores'].max():.3f}]",
              flush=True)

    del main_model, main_tok; gc.collect(); torch.cuda.empty_cache()

    # ---- Phase 3: intent route (load intent adapter, score intent subsets) ----
    print("\n=== Intent route ===")
    intent_adapter = str(BUNDLED_INTENT) if BUNDLED_INTENT.exists() else INTENT_ADAPTER
    print(f"  adapter={intent_adapter}")
    intent_model, intent_tok = load_judge(intent_adapter)
    intent_lids = binary_token_ids(intent_tok)
    print(f"  label_ids={intent_lids}")

    for ds_name, data in all_data.items():
        intent_pos = np.flatnonzero(data["intent_f"]).tolist()
        if not intent_pos:
            data["intent_scores"] = np.array([], dtype=float)
            print(f"  {data['short']}: no intent rows")
            continue
        t0 = time.perf_counter()
        prompts = [build_intent_empty_prompt(data["messages"][p], intent_tok)
                   for p in intent_pos]
        data["intent_scores"] = score_prompts(intent_model, intent_tok, intent_lids, prompts)
        print(f"  {data['short']}: {len(intent_pos)} intent rows in "
              f"{time.perf_counter()-t0:.1f}s", flush=True)

    del intent_model, intent_tok; gc.collect(); torch.cuda.empty_cache()

    # ---- Phase 4: blend and compute AUROC ----
    print("\n=== Blend + AUROC ===")
    all_aurocs, per_dataset = [], {}
    cache = {}

    for ds_name, data in all_data.items():
        scores = blend_intent_route_scores(
            data["base_scores"], data["intent_scores"], data["intent_f"])
        data["scores"] = scores

        cache[ds_name] = [
            dict(index=data["indices"][i], label=bool(data["labels"][i]),
                 score=float(scores[i]), base_score=float(data["base_scores"][i]),
                 harm=bool(data["harm_f"][i]), action=bool(data["action_f"][i]),
                 intent=bool(data["intent_f"][i]))
            for i in range(data["n"])]

        if len(np.unique(data["labels"])) > 1:
            a = roc_auc_score(data["labels"], scores)
            all_aurocs.append(a)
            per_dataset[data["short"]] = dict(
                auroc=a, n=data["n"],
                n_harm=int(data["harm_f"].sum()),
                n_action=int(data["action_f"].sum()),
                n_intent=int(data["intent_f"].sum()))
            print(f"  {data['short']:<62} {data['n']:>5} AUROC={a:>8.4f}  "
                  f"harm={int(data['harm_f'].sum())} act={int(data['action_f'].sum())} "
                  f"intent={int(data['intent_f'].sum())}")
        else:
            print(f"  {data['short']:<62} {data['n']:>5} single class")

    args.out.write_text(json.dumps(cache))

    if all_aurocs:
        print(f"\n=== Aggregate ===")
        print(f"Mean AUROC: {np.mean(all_aurocs):.4f}")
        print(f"Min  AUROC: {np.min(all_aurocs):.4f}")
        print(f"Max  AUROC: {np.max(all_aurocs):.4f}")

        summary = dict(
            judge="Phoenix v6.3 full multi-route",
            mean_auroc=float(np.mean(all_aurocs)),
            min_auroc=float(np.min(all_aurocs)),
            max_auroc=float(np.max(all_aurocs)),
            datasets=per_dataset)
        (HERE / "v63_full_judge_summary.json").write_text(json.dumps(summary, indent=2))
        print("Summary saved.")

    print(f"\nCache: {sum(len(v) for v in cache.values())} rows in {args.out}")


if __name__ == "__main__":
    main()
