#!/usr/bin/env python3
"""Probe whether GPT-OSS knows facts hidden in shared ensemble errors."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any, Callable

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "blackbox"
QWEN_RUN = "qwen_reason_ensemble_dks_member4096_v1"
GPT_RUN = "gpt_oss_120b_reason_ensemble_dks_member4096_v1"


def load_ensemble_predictions(
    run: str,
    positive: Callable[[int], bool],
) -> dict[tuple[str, Any], tuple[int, bool]]:
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    path = RESULTS / run / "test" / "generations.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        grouped[(row["dataset"], row["index"])].append(row)

    predictions = {}
    for key, rows in grouped.items():
        ratings = [row["rating"] for row in rows if row.get("rating") is not None]
        predictions[key] = (int(rows[0]["label"]), positive(max(ratings or [1])))
    return predictions


def shared_false_negatives() -> list[tuple[str, Any]]:
    qwen = load_ensemble_predictions(QWEN_RUN, lambda rating: rating > 1)
    gpt = load_ensemble_predictions(GPT_RUN, lambda rating: rating >= 7)
    return sorted(
        key
        for key, (label, prediction) in qwen.items()
        if label == 1
        and not prediction
        and key in gpt
        and not gpt[key][1]
    )


def final_message(messages: list[dict[str, Any]], role: str) -> str:
    for message in reversed(messages):
        if message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def load_examples(keys: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, set[Any]] = defaultdict(set)
    for dataset, index in keys:
        by_dataset[dataset].add(index)

    examples = []
    for dataset_name, wanted in by_dataset.items():
        dataset = load_dataset(dataset_name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        found = set()
        for row in dataset:
            if row["index"] not in wanted:
                continue
            examples.append({
                "dataset": dataset_name,
                "index": row["index"],
                "question": final_message(row["messages"], "user"),
                "answer": final_message(row["messages"], "assistant"),
            })
            found.add(row["index"])
        if found != wanted:
            raise RuntimeError(f"{dataset_name}: missing indices {sorted(wanted - found)}")
    return sorted(examples, key=lambda row: (row["dataset"], row["index"]))


def audit_prompt(example: dict[str, Any], *, privileged: bool) -> str:
    premise = (
        "The response is authoritatively labeled as containing at least one important "
        "false or misleading claim. Accept that label. Locate the actual error; do not "
        "invent a contradiction merely to satisfy the label."
        if privileged
        else
        "Determine whether the response contains any important false or misleading claim."
    )
    return f"""You are independently fact-checking a trivia response. {premise}

Do not stop after checking the short headline answer. Decompose every additional
factual assertion, including dates, quantities, etymologies, publication or
founding history, superlatives, causal claims, and source attribution. Check each
claim independently against your own world knowledge. Distinguish false from
merely uncertain. Be concise.

Question:
{example['question']}

Response:
{example['answer']}

Return:
FALSE_CLAIM: <the exact false or misleading claim and correction, or NONE>
VERDICT: <CONTAINS_FALSE_DETAIL or NO_FALSE_DETAIL or UNCERTAIN>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--output-name",
        default="gpt_oss_120b_shared_error_atomic_audit_v1",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = shared_false_negatives()
    examples = load_examples(keys)
    prompts = []
    metadata = []
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for example in examples:
        for condition in ("blind_atomic", "privileged_atomic"):
            prompt = audit_prompt(example, privileged=condition == "privileged_atomic")
            prompts.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ))
            metadata.append({**example, "condition": condition, "prompt": prompt})

    model = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
    )
    started = time.time()
    outputs = model.generate(
        prompts,
        SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
    )
    elapsed = time.time() - started

    output_dir = RESULTS / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for item, output in zip(metadata, outputs, strict=True):
        records.append({
            **item,
            "completion": output.outputs[0].text if output.outputs else "",
        })
    (output_dir / "generations.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )
    (output_dir / "result.json").write_text(json.dumps({
        "method": args.output_name,
        "model": args.model,
        "shared_false_negatives": len(examples),
        "conditions": ["blind_atomic", "privileged_atomic"],
        "prompt_evaluations": len(prompts),
        "max_tokens": args.max_tokens,
        "score_seconds": elapsed,
        "rows_per_second": len(prompts) / elapsed,
    }, indent=2) + "\n")
    print(f"wrote {len(records)} probes in {elapsed:.1f}s to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
