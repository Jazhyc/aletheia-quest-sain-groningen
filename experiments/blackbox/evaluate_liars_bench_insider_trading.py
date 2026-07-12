#!/usr/bin/env python3
"""Evaluate the frozen Phoenix v2.1 judge on Liars' Bench Insider Trading."""

from __future__ import annotations

from collections import Counter
import datetime as dt
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.blackbox.evaluate_liars_bench_hpkr import (
    ADAPTER,
    DATASET,
    JUDGE_PROMPT,
    MODEL,
    parse_prediction,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import binary_metrics
from submission.phoenix_wright_text import iter_messages, message_content, message_role


CONFIG = "insider-trading"
METHOD = "liars_bench_insider_trading_phoenix_v21_diagnostic_v1"
OUTPUT_DIR = ROOT / "results/blackbox" / METHOD
MESSAGE_MAX_CHARS = 1500
OUTPUT_MAX_CHARS = 3000
MAX_PROMPT_CHARS = 3000
MAX_PROMPT_TOKENS = 2048


def truncate_tail(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n[...truncated...]\n\n"
    keep = max_chars - len(marker)
    return text[-max_chars:] if keep <= 0 else marker + text[-keep:]


def truncate_head_tail(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n[...truncated...]\n\n"
    available = max_chars - len(marker)
    if available <= 0:
        return text[-max_chars:]
    head_chars = max(1, available // 3)
    return text[:head_chars] + marker + text[-(available - head_chars):]


def final_assistant(messages: Any) -> str:
    for message in reversed(iter_messages(messages)):
        if message_role(message) == "assistant":
            return truncate_head_tail(message_content(message), OUTPUT_MAX_CHARS)
    return ""


def notebook_user_content(messages: Any) -> tuple[str, dict[str, bool]]:
    dialogue = iter_messages(messages)
    raw_lines = [
        f"{message_role(message).upper()}: {message_content(message)}"
        for message in dialogue
    ]
    limited_lines = [
        f"{truncate_head_tail(message_role(message), 80).upper()}: "
        f"{truncate_head_tail(message_content(message), MESSAGE_MAX_CHARS)}"
        for message in dialogue
    ]
    raw_context = "\n\n".join(raw_lines)
    limited_context = "\n\n".join(limited_lines)
    raw_output = next(
        (message_content(message) for message in reversed(dialogue) if message_role(message) == "assistant"),
        "",
    )
    context = truncate_tail(limited_context, MAX_PROMPT_CHARS)
    output = final_assistant(dialogue)
    content = (
        "<context>\n" + context + "\n</context>\n\n"
        "<output>\n" + output + "\n</output>"
    )
    flags = {
        "message_char_truncated": any(len(message_content(message)) > MESSAGE_MAX_CHARS for message in dialogue),
        "context_char_truncated": len(limited_context) > MAX_PROMPT_CHARS,
        "output_char_truncated": len(raw_output) > OUTPUT_MAX_CHARS,
        "raw_context_over_5000_chars": len(raw_context) > 5000,
    }
    return content, flags


def percentile_summary(values: list[int]) -> dict[str, int]:
    return {
        "p50": int(np.percentile(values, 50)),
        "p95": int(np.percentile(values, 95)),
        "p99": int(np.percentile(values, 99)),
        "max": int(max(values)),
    }


def load_records(tokenizer: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET, data_dir=CONFIG, split="test")
    rows: list[dict[str, Any]] = []
    raw_tokens: list[int] = []
    formatted_tokens: list[int] = []
    flag_counts: Counter[str] = Counter()
    for row in dataset:
        dialogue = list(iter_messages(row["messages"]))
        raw_chat = tokenizer.apply_chat_template(
            dialogue, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        content, flags = notebook_user_content(dialogue)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": JUDGE_PROMPT + "\n\n" + content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw_ids = tokenizer(raw_chat, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        kept_ids = prompt_ids[-MAX_PROMPT_TOKENS:]
        raw_tokens.append(len(raw_ids))
        formatted_tokens.append(len(prompt_ids))
        flag_counts.update(name for name, active in flags.items() if active)
        rows.append({
            "index": row["index"],
            "source_model": row["model"],
            "source_dataset": row.get("dataset", CONFIG),
            "meta": row.get("meta", ""),
            "label": int(row["deceptive"]),
            "prompt_token_ids": kept_ids,
            "raw_conversation_tokens": len(raw_ids),
            "formatted_prompt_tokens": len(prompt_ids),
            "token_truncated": len(prompt_ids) > MAX_PROMPT_TOKENS,
            "retained_fraction": min(1.0, MAX_PROMPT_TOKENS / len(prompt_ids)),
            **flags,
        })
    length_summary = {
        "raw_conversation_tokens": percentile_summary(raw_tokens),
        "notebook_formatted_prompt_tokens_before_token_truncation": percentile_summary(formatted_tokens),
        "token_truncated_rows": sum(length > MAX_PROMPT_TOKENS for length in formatted_tokens),
        "token_truncated_fraction": float(np.mean(np.array(formatted_tokens) > MAX_PROMPT_TOKENS)),
        "character_truncation_counts": dict(flag_counts),
    }
    return pd.DataFrame(rows), length_summary


def evaluate(llm: Any, sampling: Any, records: pd.DataFrame, lora_request: Any) -> tuple[pd.DataFrame, float]:
    from vllm.inputs import TokensPrompt

    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in records["prompt_token_ids"]]
    started = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)
    elapsed = time.time() - started
    evaluated = records.drop(columns="prompt_token_ids").copy()
    generations = [output.outputs[0].text if output.outputs else "" for output in outputs]
    predictions = [parse_prediction(text) for text in generations]
    evaluated["prediction"] = predictions
    evaluated["score"] = [float(value) if value is not None else 0.0 for value in predictions]
    evaluated["parse_error"] = [value is None for value in predictions]
    evaluated["generation"] = generations
    return evaluated, elapsed


def summarize(frame: pd.DataFrame, elapsed: float) -> dict[str, Any]:
    def metrics(group: pd.DataFrame) -> dict[str, Any]:
        return binary_metrics(group["label"].to_numpy(), group["score"].to_numpy(), 0.5)

    return {
        "rows": len(frame),
        "labels": dict(Counter(frame["label"].astype(str))),
        "metrics": metrics(frame),
        "per_source_model": {
            model: metrics(group) for model, group in frame.groupby("source_model", sort=True)
        },
        "by_token_truncation": {
            str(truncated): metrics(group)
            for truncated, group in frame.groupby("token_truncated", sort=True)
        },
        "parse_errors": int(frame["parse_error"].sum()),
        "score_seconds": elapsed,
        "rows_per_second": len(frame) / elapsed,
    }


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    records, length_summary = load_records(tokenizer)
    llm = LLM(
        model=MODEL,
        tokenizer=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        max_num_seqs=256,
        enable_lora=True,
        max_lora_rank=16,
    )
    sampling = SamplingParams(max_tokens=512, temperature=0.0)
    conditions = {
        "base": None,
        "phoenix_v21_adapter": LoRARequest("phoenix-v21", 1, ADAPTER.as_posix()),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "method": METHOD,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": DATASET,
        "config": CONFIG,
        "judge_model": MODEL,
        "include_reasoning": False,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "lengths": length_summary,
        "conditions": {},
    }
    for name, request in conditions.items():
        frame, elapsed = evaluate(llm, sampling, records, request)
        frame.to_json(OUTPUT_DIR / f"{name}_generations.jsonl", orient="records", lines=True)
        result["conditions"][name] = summarize(frame, elapsed)
        print(name, result["conditions"][name], flush=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
