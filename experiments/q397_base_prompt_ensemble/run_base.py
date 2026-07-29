#!/usr/bin/env python3
"""Score the frozen explicit-digit base-Qwen prompt on local test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from experiments.q397_family_calibration.cache_logits import load_records
from experiments.q397_readout_adaptation.run import (
    MODEL_ID,
    requested_logprobs,
    sigmoid,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "results/blackbox/q397_base_explicit_ensemble_test_base_v1"
)
DIGIT_IDS = [15, 16]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one no-LoRA direct-logit pass with the frozen base prompt."""
    from experiments.phoenix_verbalizer_sweep.run_ndif_base_verbalizers import (
        CONDITIONS,
        build_direct_prompt,
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    records = load_records(args.split, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    condition = next(item for item in CONDITIONS if item.name == "digits_explicit")
    prompts = [
        build_direct_prompt(record["messages"], tokenizer, condition)
        for record in records
    ]
    llm = LLM(
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
    )
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=2,
        logprob_token_ids=DIGIT_IDS,
        allowed_token_ids=DIGIT_IDS,
    )
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - started
    logprobs = [requested_logprobs(output, DIGIT_IDS) for output in outputs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scores.jsonl").open("w") as handle:
        for record, prompt, values in zip(
            records, prompts, logprobs, strict=True
        ):
            margin = values[DIGIT_IDS[1]] - values[DIGIT_IDS[0]]
            handle.write(
                json.dumps(
                    {
                        "dataset": record["dataset"],
                        "index": record["index"],
                        "label": record["label"],
                        "score": sigmoid(margin),
                        "logit_margin": margin,
                        "prompt_sha256": hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    report = {
        "method": "q397_base_explicit_ensemble_test_base_v1",
        "split": args.split,
        "rows": len(records),
        "model": MODEL_ID,
        "adapter": None,
        "condition": condition.name,
        "score_seconds": elapsed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
