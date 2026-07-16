#!/usr/bin/env python3
"""Measure continuous neutral TRUE/FALSE margins from each generating Qwen organism."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fact_extractor_self_consistency.run_qwen_whole_output_self_query import (
    MODEL,
    PROMPTS,
    load_qwen_validation,
    render,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import macro_metrics


METHOD = "qwen27b_whole_output_self_query_margin_validation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/blackbox" / METHOD)
    return parser.parse_args()


def single_token_id(tokenizer: Any, text: str) -> int:
    """Require the constrained verdict to be one tokenizer token."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not a single token: {ids}")
    return int(ids[0])


def logprob_value(value: Any) -> float:
    """Read a vLLM Logprob or a numeric test double."""
    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def false_score_from_logprobs(
    first_token_logprobs: dict[Any, Any],
    true_token_id: int,
    false_token_id: int,
) -> float | None:
    """Return normalized P(FALSE) over the two constrained verdict tokens."""
    expanded = {int(key): logprob_value(value) for key, value in first_token_logprobs.items()}
    if true_token_id not in expanded or false_token_id not in expanded:
        return None
    difference = max(-80.0, min(80.0, expanded[false_token_id] - expanded[true_token_id]))
    return 1.0 / (1.0 + math.exp(-difference))


def score_outputs(
    outputs: list[Any],
    true_token_id: int,
    false_token_id: int,
) -> list[tuple[float, str, bool]]:
    """Extract scores, chosen completions, and missing-logprob flags."""
    rows = []
    for output in outputs:
        completion = output.outputs[0].text if output.outputs else ""
        first = {}
        if output.outputs and output.outputs[0].logprobs:
            first = output.outputs[0].logprobs[0] or {}
        score = false_score_from_logprobs(first, true_token_id, false_token_id)
        missing = score is None
        rows.append((0.5 if score is None else score, completion, missing))
    return rows


def main() -> None:
    args = parse_args()
    records = load_qwen_validation()
    if args.limit is not None:
        records = records[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    true_token_id = single_token_id(tokenizer, "TRUE")
    false_token_id = single_token_id(tokenizer, "FALSE")
    verdict_ids = [true_token_id, false_token_id]

    lora_repos = sorted({str(row["lora"]) for row in records if row["lora"]})
    lora_paths = {repo: Path(snapshot_download(repo_id=repo)) for repo in lora_repos}
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_lora=True,
        max_lora_rank=128,
        max_loras=1,
        max_cpu_loras=max(1, len(lora_repos)),
    )
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(verdict_ids),
        logprob_token_ids=verdict_ids,
        allowed_token_ids=verdict_ids,
    )

    groups: list[tuple[str | None, list[dict[str, Any]]]] = [
        (None, [row for row in records if not row["lora"]])
    ] + [(repo, [row for row in records if row["lora"] == repo]) for repo in lora_repos]
    generations: list[dict[str, Any]] = []
    started = time.time()
    for lora_id, (repo, group) in enumerate(groups):
        if not group:
            continue
        request = None if repo is None else LoRARequest(repo, lora_id, lora_paths[repo].as_posix())
        for variant, template in PROMPTS.items():
            prompts = [render(tokenizer, template.format(output=row["output"])) for row in group]
            outputs = llm.generate(prompts, sampling, lora_request=request)
            scored = score_outputs(outputs, true_token_id, false_token_id)
            for row, prompt, (score, completion, missing) in zip(
                group, prompts, scored, strict=True
            ):
                generations.append({
                    **row,
                    "variant": variant,
                    "prompt": prompt,
                    "completion": completion,
                    "false_score": score,
                    "missing_logprobs": missing,
                })
    elapsed = time.time() - started

    results = []
    for variant in PROMPTS:
        rows = [row for row in generations if row["variant"] == variant]
        frame = pd.DataFrame({
            "dataset": [row["dataset"] for row in rows],
            "label": [row["label"] for row in rows],
            "score": [row["false_score"] for row in rows],
        })
        results.append({
            "variant": variant,
            "rows": len(rows),
            "metrics": macro_metrics(frame, 0.5),
            "missing_logprobs": sum(row["missing_logprobs"] for row in rows),
            "chosen_tokens": dict(Counter(row["completion"].strip() for row in rows)),
        })

    summary = {
        "method": args.output_dir.name,
        "model": MODEL,
        "split": "validation",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": len(records),
        "lora_repos": lora_repos,
        "prompt_evaluations": len(generations),
        "score_seconds": elapsed,
        "evaluations_per_second": len(generations) / elapsed if elapsed else None,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "generations.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in generations) + "\n"
    )
    (args.output_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
