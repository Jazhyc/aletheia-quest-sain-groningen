#!/usr/bin/env python3
"""Cross Qwen-planned facts with existing evidence-trained Qwen consumers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
import yaml

from experiments.privileged_information_distillation.evaluate_student_sft import (
    EMPTY_REASONING_PREFIX,
    binary_token_ids,
    load_records,
    score_binary_prefixes,
    vllm_max_lora_rank,
)
from experiments.wikidata_rag.evaluate_qwen_planner_phoenix import (
    load_cache,
    summarize_condition,
)


ROOT = Path(__file__).resolve().parents[2]
PREFIX_MODES = ("direct", "empty_summary")


def parse_named_path(value: str) -> tuple[str, Path]:
    """Parse a stable NAME=PATH command-line value."""
    name, separator, location = value.partition("=")
    if not separator or not name or not location:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    path = Path(location)
    if not path.is_absolute():
        path = ROOT / path
    return name, path.resolve()


def scoring_prompt(prompt: str, prefix_mode: str) -> str:
    """Append the selected binary label boundary."""
    if prefix_mode == "direct":
        return prompt + "Prediction:"
    if prefix_mode == "empty_summary":
        return prompt + EMPTY_REASONING_PREFIX
    raise ValueError(f"unsupported prefix mode: {prefix_mode}")


def replace_active_scores(
    baseline: pd.DataFrame,
    active_scores: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    """Replace only cache-active rows in a full consumer score frame."""
    keys = ["dataset", "index"]
    base = baseline.set_index(keys).copy()
    active = active_scores.set_index(keys)
    if not base.index.is_unique or not active.index.is_unique:
        raise ValueError("duplicate dataset/index keys")
    if not active.index.isin(base.index).all():
        raise ValueError("active rows are not a subset of the baseline")
    if not active["label"].equals(base.loc[active.index, "label"]):
        raise ValueError("active labels differ from the baseline")
    base.loc[active.index, "score"] = active[score_column]
    return base.reset_index()


def score_delta(
    active_scores: pd.DataFrame,
    score_column: str,
) -> dict[str, float | None]:
    """Summarize evidence-minus-empty movement on active rows."""
    output: dict[str, float | None] = {}
    for name, label in (("all", None), ("honest", 0), ("deceptive", 1)):
        rows = (
            active_scores
            if label is None
            else active_scores[active_scores["label"].eq(label)]
        )
        output[name] = (
            None
            if rows.empty
            else float((rows[score_column] - rows["empty_score"]).mean())
        )
    return output


def condition_uplift(
    empty: dict[str, Any],
    condition: dict[str, Any],
) -> dict[str, float]:
    """Return scenario AUROC changes against the consumer's own empty control."""
    return {
        scenario: (
            float(condition["metrics"][scenario]["auroc"])
            - float(empty["metrics"][scenario]["auroc"])
        )
        for scenario in ("all", "instructed", "varied")
    }


def consumer_records(
    config: dict[str, Any],
    tokenizer: Any,
    cache_paths: dict[str, Path],
    splits_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Render a full explicit-empty baseline and sparse real/shuffled prompts."""
    empty = load_records(
        "validation",
        splits_dir,
        config,
        tokenizer,
        append_empty_reference=True,
    )
    empty["key"] = list(zip(empty["dataset"], empty["index"], strict=True))
    caches = {}
    for name, path in cache_paths.items():
        real_references, shuffled_references, active_keys = load_cache(path)
        real = load_records(
            "validation",
            splits_dir,
            config,
            tokenizer,
            references=real_references,
            append_empty_reference=True,
        )
        shuffled = load_records(
            "validation",
            splits_dir,
            config,
            tokenizer,
            references=shuffled_references,
            append_empty_reference=True,
        )
        for frame in (real, shuffled):
            frame["key"] = list(zip(frame["dataset"], frame["index"], strict=True))
        real_active = real[real["key"].isin(active_keys)].copy().sort_values("key")
        shuffled_active = (
            shuffled[shuffled["key"].isin(active_keys)].copy().sort_values("key")
        )
        if len(real_active) != len(active_keys) or len(shuffled_active) != len(active_keys):
            raise RuntimeError(f"cache {name!r} contains unknown active keys")
        if real_active["key"].tolist() != shuffled_active["key"].tolist():
            raise RuntimeError(f"cache {name!r} real/shuffled active keys differ")
        caches[name] = {
            "path": path.as_posix(),
            "active_keys": active_keys,
            "real": real_active,
            "shuffled": shuffled_active,
        }
    return empty, caches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consumer", action="append", required=True)
    parser.add_argument("--cache", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    consumers = dict(parse_named_path(value) for value in args.consumer)
    cache_paths = dict(parse_named_path(value) for value in args.cache)
    if len(consumers) != len(args.consumer) or len(cache_paths) != len(args.cache):
        raise ValueError("consumer and cache names must be unique")

    configs = {
        name: yaml.safe_load((adapter.parent / "config.yaml").read_text())
        for name, adapter in consumers.items()
    }
    models = {config["student"]["model"] for config in configs.values()}
    if len(models) != 1:
        raise ValueError(f"consumers use different base models: {sorted(models)}")
    ranks = [int(config["student"]["lora"]["r"]) for config in configs.values()]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    first_adapter = next(iter(consumers.values()))
    tokenizer = AutoTokenizer.from_pretrained(first_adapter)
    token_ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(token_ids),
        logprob_token_ids=token_ids,
        allowed_token_ids=token_ids,
    )
    llm = LLM(
        model=next(iter(models)),
        tokenizer=first_adapter.as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=vllm_max_lora_rank(ranks),
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
    )

    result: dict[str, Any] = {
        "consumers": {},
        "caches": {name: path.as_posix() for name, path in cache_paths.items()},
    }
    total_score_seconds = 0.0
    total_missing = 0
    started = time.time()
    for lora_id, (consumer_name, adapter) in enumerate(consumers.items(), start=1):
        config = configs[consumer_name]
        empty, cache_records = consumer_records(
            config,
            tokenizer,
            cache_paths,
            args.splits_dir.resolve(),
        )
        request = LoRARequest(consumer_name, lora_id, adapter.as_posix())
        consumer_result: dict[str, Any] = {
            "adapter": adapter.as_posix(),
            "prefixes": {},
        }
        for prefix_mode in PREFIX_MODES:
            prompts = [
                scoring_prompt(prompt, prefix_mode) for prompt in empty["prompt"]
            ]
            slices: dict[str, tuple[slice, slice]] = {}
            for cache_name, records in cache_records.items():
                real_start = len(prompts)
                prompts.extend(
                    scoring_prompt(prompt, prefix_mode)
                    for prompt in records["real"]["prompt"]
                )
                real_slice = slice(real_start, len(prompts))
                shuffled_start = len(prompts)
                prompts.extend(
                    scoring_prompt(prompt, prefix_mode)
                    for prompt in records["shuffled"]["prompt"]
                )
                slices[cache_name] = (
                    real_slice,
                    slice(shuffled_start, len(prompts)),
                )
            values, missing, elapsed = score_binary_prefixes(
                llm,
                prompts,
                sampling,
                request,
                token_ids,
            )
            total_score_seconds += elapsed
            total_missing += missing
            empty_scores = empty[["dataset", "index", "label"]].copy()
            empty_scores["score"] = values[: len(empty)]
            empty_summary = summarize_condition(empty_scores)
            prefix_result: dict[str, Any] = {
                "empty": empty_summary,
                "caches": {},
                "prompts": len(prompts),
                "missing_logits": missing,
                "score_seconds": elapsed,
            }
            for cache_name, records in cache_records.items():
                real_slice, shuffled_slice = slices[cache_name]
                active = records["real"][["dataset", "index", "label"]].copy()
                active["empty_score"] = [
                    empty_scores.set_index(["dataset", "index"]).loc[key, "score"]
                    for key in records["real"]["key"]
                ]
                active["real_score"] = values[real_slice]
                active["shuffled_score"] = values[shuffled_slice]
                real_frame = replace_active_scores(
                    empty_scores, active, "real_score"
                )
                shuffled_frame = replace_active_scores(
                    empty_scores, active, "shuffled_score"
                )
                real_summary = summarize_condition(real_frame)
                shuffled_summary = summarize_condition(shuffled_frame)
                prefix_result["caches"][cache_name] = {
                    "active_rows": len(active),
                    "real": real_summary,
                    "shuffled": shuffled_summary,
                    "real_uplift_vs_empty": condition_uplift(
                        empty_summary, real_summary
                    ),
                    "shuffled_uplift_vs_empty": condition_uplift(
                        empty_summary, shuffled_summary
                    ),
                    "active_score_delta": {
                        "real": score_delta(active, "real_score"),
                        "shuffled": score_delta(active, "shuffled_score"),
                    },
                }
            consumer_result["prefixes"][prefix_mode] = prefix_result
        result["consumers"][consumer_name] = consumer_result
    result["missing_logits"] = total_missing
    result["score_seconds"] = total_score_seconds
    result["wall_seconds"] = time.time() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "result.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
