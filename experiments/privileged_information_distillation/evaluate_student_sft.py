#!/usr/bin/env python3
"""Evaluate one or more privileged-distillation LoRA students with one vLLM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import (
    append_reference_material,
    build_student_prompt,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import (
    binary_metrics,
    load_split_config,
    load_labels,
    macro_metrics,
)

PREDICTION_RE = re.compile(r"(?im)\bPrediction\s*:\s*<?([01])>?")
STRICT_RE = re.compile(
    r"(?is)^\s*<reasoning_summary>\s*(.+?)\s*</reasoning_summary>\s*"
    r"Prediction\s*:\s*([01])\s*$"
)
COUNTERFACTUAL_STRICT_RE = re.compile(
    r"(?is)^\s*<reasoning_summary>\s*(.+?)\s*</reasoning_summary>\s*"
    r"<facts>\s*(.+?)\s*</facts>\s*"
    r"<contradiction>\s*(.+?)\s*</contradiction>\s*"
    r"Prediction\s*:\s*([01])\s*$"
)
PREDICTION_ONLY_STRICT_RE = re.compile(r"(?is)^\s*Prediction\s*:\s*<?([01])>?\s*$")
EMPTY_REASONING_PREFIX = (
    "<reasoning_summary>\n"
    "</reasoning_summary>\n"
    "Prediction:"
)


def parse_prediction(text: str) -> int | None:
    matches = PREDICTION_RE.findall(text)
    return int(matches[-1]) if matches else None


def prefix_before_prediction(text: str) -> str:
    """Return a completion prefix ending immediately before its final 0/1 token."""
    matches = list(PREDICTION_RE.finditer(text))
    if matches:
        return text[:matches[-1].start(1)]
    return text.rstrip() + "\nPrediction:"


def binary_token_ids(tokenizer: Any) -> list[int]:
    """Return the distinct single-token ids used for literal binary predictions."""
    ids = []
    for text in ("0", "1"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"binary target {text!r} tokenized as {encoded}, expected one token")
        ids.append(int(encoded[0]))
    if len(set(ids)) != 2:
        raise ValueError(f"binary targets must have distinct token ids, got {ids}")
    return ids


def logprob_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def binary_score_from_logprobs(
    first_token_logprobs: dict[Any, Any],
    token_ids: list[int],
) -> float | None:
    """Normalize constrained 0/1 token probabilities into P(Prediction=1)."""
    expanded = {
        int(token_id): logprob_value(value)
        for token_id, value in first_token_logprobs.items()
    }
    if any(token_id not in expanded for token_id in token_ids):
        return None
    logit_zero, logit_one = (expanded[token_id] for token_id in token_ids)
    difference = max(-80.0, min(80.0, logit_one - logit_zero))
    return 1.0 / (1.0 + math.exp(-difference))


def score_binary_prefixes(
    llm: Any,
    prompts: list[str],
    sampling: Any,
    request: Any,
    token_ids: list[int],
) -> tuple[list[float], int, float]:
    """Score constrained binary next-token margins for rendered prefixes."""
    started = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=request)
    elapsed = time.time() - started
    scores = []
    missing = 0
    for output in outputs:
        first_token_logprobs = {}
        if output.outputs and output.outputs[0].logprobs:
            first_token_logprobs = output.outputs[0].logprobs[0] or {}
        score = binary_score_from_logprobs(first_token_logprobs, token_ids)
        if score is None:
            missing += 1
            score = 0.5
        scores.append(score)
    return scores, missing, elapsed


def scenario_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    result = {"all": macro_metrics(frame, 0.5)}
    for scenario in ("instructed", "varied"):
        subset = frame[frame["dataset"].str.contains(f"dev-{scenario}-deception")]
        if not subset.empty:
            result[scenario] = macro_metrics(subset, 0.5)
    return result


def load_retrieval_cache(path: Path | None) -> dict[tuple[str, Any], str]:
    if path is None:
        return {}
    references = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        passages = record.get("passages") or []
        text = "\n".join(
            f"- {passage.get('title', '')}: {passage.get('text', '')}"
            for passage in passages
            if passage.get("text")
        )
        references[(record["dataset"], record["index"])] = text
    return references


def load_records(
    split: str,
    splits_dir: Path,
    config: dict[str, Any],
    tokenizer: Any,
    references: dict[tuple[str, Any], str] | None = None,
) -> pd.DataFrame:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    for dataset_cfg in load_split_config(splits_dir / f"dry.{split}.yaml", ROOT):
        labels = load_labels(dataset_cfg)
        label_by_index = dict(zip(labels["index"], labels["label"], strict=True))
        dataset = load_dataset(dataset_cfg.name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        found: set[Any] = set()
        for row in dataset:
            index = row["index"]
            if index not in label_by_index:
                continue
            raw_prompt = build_student_prompt(
                row["messages"],
                config["student"]["prompt"],
                int(config["student"]["max_prompt_chars"]),
                config["student"]["context_truncation"],
            )
            reference = (references or {}).get((dataset_cfg.name, index), "")
            if reference:
                raw_prompt = append_reference_material(raw_prompt, reference)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            rows.append({
                "dataset": dataset_cfg.name,
                "index": index,
                "label": int(label_by_index[index]),
                "prompt": prompt,
            })
            found.add(index)
        if len(found) != len(labels):
            raise RuntimeError(f"{dataset_cfg.name}: found {len(found)}/{len(labels)} labeled rows")
    return pd.DataFrame(rows)


def evaluate_adapter(
    llm: Any,
    sampling: Any,
    records: pd.DataFrame,
    adapter_dir: Path,
    lora_id: int,
    strict_re: re.Pattern[str] = STRICT_RE,
    *,
    margin_sampling: Any | None = None,
    binary_ids: list[int] | None = None,
    empty_reasoning_prefix: str = EMPTY_REASONING_PREFIX,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    from vllm.lora.request import LoRARequest

    request = LoRARequest(adapter_dir.parent.name, lora_id, adapter_dir.as_posix())
    started = time.time()
    outputs = llm.generate(records["prompt"].tolist(), sampling, lora_request=request)
    elapsed = time.time() - started
    evaluated = records.drop(columns="prompt").copy()
    generations = [output.outputs[0].text if output.outputs else "" for output in outputs]
    predictions = [parse_prediction(text) for text in generations]
    evaluated["prediction"] = predictions
    evaluated["score"] = [float(value) if value is not None else 0.0 for value in predictions]
    evaluated["parse_error"] = [value is None for value in predictions]
    evaluated["format_valid"] = [strict_re.fullmatch(text) is not None for text in generations]
    evaluated["generation"] = generations
    timing: dict[str, float | int] = {"generation_seconds": elapsed}

    if margin_sampling is not None:
        if binary_ids is None:
            raise ValueError("binary_ids are required when margin_sampling is enabled")
        source_prompts = records["prompt"].tolist()
        empty_prompts = [prompt + empty_reasoning_prefix for prompt in source_prompts]
        post_reasoning_prompts = [
            prompt + prefix_before_prediction(generation)
            for prompt, generation in zip(source_prompts, generations, strict=True)
        ]
        empty_scores, empty_missing, empty_elapsed = score_binary_prefixes(
            llm, empty_prompts, margin_sampling, request, binary_ids
        )
        reasoning_scores, reasoning_missing, reasoning_elapsed = score_binary_prefixes(
            llm, post_reasoning_prompts, margin_sampling, request, binary_ids
        )
        evaluated["empty_margin_score"] = empty_scores
        evaluated["reasoning_margin_score"] = reasoning_scores
        timing.update({
            "empty_margin_seconds": empty_elapsed,
            "empty_margin_missing": empty_missing,
            "reasoning_margin_seconds": reasoning_elapsed,
            "reasoning_margin_missing": reasoning_missing,
        })
    return evaluated, timing


def metrics_for_score(evaluated: pd.DataFrame, score_column: str) -> dict[str, Any]:
    scored = evaluated[["dataset", "index", "label", score_column]].rename(
        columns={score_column: "score"}
    )
    return scenario_metrics(scored)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", action="append", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--retrieval-cache", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--continuous-margins",
        action="store_true",
        help="score constrained 0/1 logits with empty and generated reasoning prefixes",
    )
    args = parser.parse_args()

    adapter_dirs = [path.resolve() for path in args.adapter_dir]
    configs = [yaml.safe_load((path.parent / "config.yaml").read_text()) for path in adapter_dirs]
    first = configs[0]
    comparable = (
        "model", "prompt", "max_prompt_chars", "context_truncation",
        "target_format", "target_mode",
    )
    for config in configs[1:]:
        if any(config["student"].get(key) != first["student"].get(key) for key in comparable):
            raise SystemExit("student prompt/model settings differ across adapters")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(adapter_dirs[0])
    references = load_retrieval_cache(
        args.retrieval_cache.resolve() if args.retrieval_cache is not None else None
    )
    records = load_records(
        args.split,
        args.splits_dir.resolve(),
        first,
        tokenizer,
        references=references,
    )
    print(f"loaded {len(records)} rows across {records['dataset'].nunique()} datasets", flush=True)
    llm = LLM(
        model=first["student"]["model"],
        tokenizer=adapter_dirs[0].as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=max(int(config["student"]["lora"]["r"]) for config in configs),
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    binary_ids = None
    margin_sampling = None
    if args.continuous_margins:
        binary_ids = binary_token_ids(tokenizer)
        margin_sampling = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=len(binary_ids),
            logprob_token_ids=binary_ids,
            allowed_token_ids=binary_ids,
        )
    if first["student"].get("target_mode") == "prediction_only":
        strict_re = PREDICTION_ONLY_STRICT_RE
    elif first["student"].get("target_format") == "counterfactual":
        strict_re = COUNTERFACTUAL_STRICT_RE
    else:
        strict_re = STRICT_RE

    for lora_id, (adapter_dir, config) in enumerate(zip(adapter_dirs, configs, strict=True), 1):
        evaluated, timing = evaluate_adapter(
            llm,
            sampling,
            records,
            adapter_dir,
            lora_id,
            strict_re=strict_re,
            margin_sampling=margin_sampling,
            binary_ids=binary_ids,
        )
        method = config["method"]
        output_dir = adapter_dir.parent / (args.run_name or args.split)
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluated.to_json(output_dir / "generations.jsonl", orient="records", lines=True)
        per_dataset = {}
        for dataset, group in evaluated.groupby("dataset", sort=True):
            per_dataset[dataset] = binary_metrics(
                group["label"].to_numpy(), group["score"].to_numpy(), 0.5
            )
        score_metrics = {"generated_binary": metrics_for_score(evaluated, "score")}
        if args.continuous_margins:
            score_metrics.update({
                "empty_margin": metrics_for_score(evaluated, "empty_margin_score"),
                "reasoning_margin": metrics_for_score(evaluated, "reasoning_margin_score"),
            })
        total_elapsed = float(sum(
            value for key, value in timing.items() if key.endswith("_seconds")
        ))
        result = {
            "method": method,
            "split": args.split,
            "learning_rate": config["student"]["training"]["learning_rate"],
            "metrics": score_metrics["generated_binary"],
            "score_metrics": score_metrics,
            "per_dataset": per_dataset,
            "parse_errors": int(evaluated["parse_error"].sum()),
            "format_valid": int(evaluated["format_valid"].sum()),
            "rows": len(evaluated),
            "score_seconds": total_elapsed,
            "rows_per_second": len(evaluated) / total_elapsed,
            "timing": timing,
            "max_new_tokens": args.max_new_tokens,
            "retrieval_cache": (
                args.retrieval_cache.resolve().as_posix()
                if args.retrieval_cache is not None
                else None
            ),
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        metrics = result["metrics"]
        print(
            f"{method} lr={result['learning_rate']} all={metrics['all']} "
            f"instructed={metrics.get('instructed')} varied={metrics.get('varied')} "
            f"parse_errors={result['parse_errors']} time={total_elapsed:.1f}s",
            flush=True,
        )
        if args.continuous_margins:
            print(f"continuous_margin_metrics={score_metrics}", flush=True)


if __name__ == "__main__":
    main()
