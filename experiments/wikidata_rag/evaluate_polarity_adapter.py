#!/usr/bin/env python3
"""Evaluate direct support/contradiction margins on frozen teacher labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.privileged_information_distillation.evaluate_student_sft import (
    binary_token_ids,
    score_binary_prefixes,
)
from experiments.wikidata_rag.build_polarity_distillation import polarity_prompt


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def polar_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (row.get("annotation") or {}).get("polarity")
        in {"supports", "contradicts"}
    ]


def render(rows: list[dict[str, Any]], tokenizer: Any) -> tuple[list[str], np.ndarray]:
    prompts = []
    labels = []
    for row in rows:
        raw = polarity_prompt(row, row["candidate"])
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "Prediction:"
        )
        labels.append(int(row["annotation"]["polarity"] == "contradicts"))
    return prompts, np.asarray(labels, dtype=np.int8)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    both = len(np.unique(labels)) == 2
    return {
        "candidates": len(labels),
        "contradictions": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)) if both else None,
        "average_precision": (
            float(average_precision_score(labels, scores)) if both else None
        ),
        "mean_support_score": (
            float(scores[labels == 0].mean()) if bool((labels == 0).any()) else None
        ),
        "mean_contradiction_score": (
            float(scores[labels == 1].mean()) if bool((labels == 1).any()) else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = polar_rows(load(args.validation_input))
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    prompts, labels = render(rows, tokenizer)
    token_ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(token_ids),
        logprob_token_ids=token_ids,
        allowed_token_ids=token_ids,
    )
    llm = LLM(
        model="Qwen/Qwen3.5-9B",
        tokenizer=args.adapter_dir.as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enable_lora=True,
        max_lora_rank=4,
        max_model_len=2048,
        max_num_seqs=args.batch_size,
    )
    request = LoRARequest(args.adapter_dir.parent.name, 1, args.adapter_dir.as_posix())
    started = time.time()
    base_scores, base_missing, base_seconds = score_binary_prefixes(
        llm, prompts, sampling, None, token_ids
    )
    adapter_scores, adapter_missing, adapter_seconds = score_binary_prefixes(
        llm, prompts, sampling, request, token_ids
    )
    base_scores = np.asarray(base_scores)
    adapter_scores = np.asarray(adapter_scores)
    report = {
        "validation_input": args.validation_input.as_posix(),
        "base": metrics(labels, base_scores),
        "adapter": metrics(labels, adapter_scores),
        "missing_logits": {"base": base_missing, "adapter": adapter_missing},
        "score_seconds": {"base": base_seconds, "adapter": adapter_seconds},
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
