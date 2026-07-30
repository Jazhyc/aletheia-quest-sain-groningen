#!/usr/bin/env python3
"""Score Kimi training prompts with the distilled adapter and matched base."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_METHOD = "kimi_k3_student_train_imitation_audit_v1"
DEFAULT_CACHE = (
    ROOT
    / "results/blackbox/kimi_k3_fireworks_nothink_tvg_binary_logit_v1/train"
)
DEFAULT_ADAPTER = (
    ROOT
    / "results/blackbox/qwen9b_kimi_k3_openrouter_tvg_soft_r16_lr5e5_ep2_v1/"
    "adapter"
)


def keyed_jsonl(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load unique dataset/index rows from JSONL."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        row = json.loads(line)
        key = (str(row["dataset"]), str(row["index"]))
        if key in rows:
            raise ValueError(f"duplicate key at {path}:{line_number}: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"empty JSONL artifact: {path}")
    return rows


def load_training_rows(
    student_rows_path: Path,
    soft_targets_path: Path,
) -> list[dict[str, Any]]:
    """Join exact cached prompts to their Kimi probability and margin."""
    prompts = keyed_jsonl(student_rows_path)
    targets = keyed_jsonl(soft_targets_path)
    if prompts.keys() != targets.keys():
        raise ValueError(
            f"cache keys differ: prompts={len(prompts)} targets={len(targets)} "
            f"matched={len(prompts.keys() & targets.keys())}"
        )
    rows = []
    for key in prompts:
        prompt = prompts[key]
        target = targets[key]
        if int(prompt["label"]) != int(target["label"]):
            raise ValueError(f"label mismatch for {key}")
        probability = float(target["soft_target"])
        if not 0.0 < probability < 1.0:
            raise ValueError(f"invalid teacher probability for {key}: {probability}")
        serialized_margin = float(target["teacher_logit"])
        reconstructed_margin = math.log(probability) - math.log1p(-probability)
        if not math.isclose(
            serialized_margin,
            reconstructed_margin,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise ValueError(f"teacher margin/probability mismatch for {key}")
        rows.append({
            "dataset": key[0],
            "index": prompt["index"],
            "label": int(prompt["label"]),
            "raw_prompt": str(prompt["student_prompt"]),
            "teacher_probability": probability,
            "teacher_margin": serialized_margin,
        })
    return rows


def resolve_binary_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Resolve literal 0/1 labels to distinct single token ids."""
    token_ids = []
    for text in ("0", "1"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"label {text!r} tokenized as {encoded}")
        token_ids.append(int(encoded[0]))
    if token_ids[0] == token_ids[1]:
        raise ValueError("binary labels resolved to the same token")
    return token_ids[0], token_ids[1]


def logprob_value(value: Any) -> float:
    """Normalize vLLM float and Logprob wrappers."""
    return float(value.logprob if hasattr(value, "logprob") else value)


def requested_logprobs(output: Any, token_ids: tuple[int, int]) -> dict[int, float]:
    """Extract both requested next-token log probabilities."""
    if not output.outputs or not output.outputs[0].logprobs:
        raise ValueError("vLLM output is missing next-token logprobs")
    values = output.outputs[0].logprobs[0]
    result = {}
    for token_id in token_ids:
        value = values.get(token_id)
        if value is None:
            value = values.get(str(token_id))
        if value is None:
            raise ValueError(f"missing requested token id {token_id}")
        result[token_id] = logprob_value(value)
    return result


def sigmoid(value: float) -> float:
    """Return a numerically stable logistic score."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exponent = math.exp(max(value, -60.0))
    return exponent / (1.0 + exponent)


def prompt_for_scoring(raw_prompt: str, tokenizer: Any) -> str:
    """Reproduce the training direct-input boundary exactly."""
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return chat + "Prediction:"


def score_outputs(
    outputs: list[Any],
    token_ids: tuple[int, int],
) -> tuple[list[float], list[float]]:
    """Convert vLLM outputs into normalized scores and raw margins."""
    margins = []
    for output in outputs:
        values = requested_logprobs(output, token_ids)
        margins.append(values[token_ids[1]] - values[token_ids[0]])
    return [sigmoid(value) for value in margins], margins


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Score the cached training prompts in one shared vLLM session."""
    from experiments.privileged_information_distillation.evaluate_student_sft import (
        validate_qwen35_adapter_layout,
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_dir = args.adapter_dir.resolve()
    validate_qwen35_adapter_layout(adapter_dir, MODEL_ID)
    rows = load_training_rows(
        args.cache_dir / "student_rows.jsonl",
        args.cache_dir / "soft_targets.jsonl",
    )
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    token_ids = resolve_binary_token_ids(tokenizer)
    prompts = [
        prompt_for_scoring(row["raw_prompt"], tokenizer) for row in rows
    ]
    prompt_hashes = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts
    ]

    llm = LLM(
        model=MODEL_ID,
        tokenizer=adapter_dir.as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=4096,
    )
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=2,
        logprob_token_ids=list(token_ids),
        allowed_token_ids=list(token_ids),
    )
    lora_request = LoRARequest(
        adapter_dir.parent.name,
        1,
        adapter_dir.as_posix(),
    )

    started = time.perf_counter()
    adapter_outputs = llm.generate(
        prompts,
        sampling,
        lora_request=lora_request,
    )
    adapter_seconds = time.perf_counter() - started
    adapter_scores, adapter_margins = score_outputs(adapter_outputs, token_ids)

    started = time.perf_counter()
    base_outputs = llm.generate(prompts, sampling)
    base_seconds = time.perf_counter() - started
    base_scores, base_margins = score_outputs(base_outputs, token_ids)

    absolute_differences = [
        abs(adapter - base)
        for adapter, base in zip(adapter_scores, base_scores, strict=True)
    ]
    if not any(value > 1.0e-7 for value in absolute_differences):
        raise RuntimeError("LoRA effect fingerprint failed")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scores.jsonl").open("w") as handle:
        for position, row in enumerate(rows):
            output = {
                key: value for key, value in row.items() if key != "raw_prompt"
            }
            output.update({
                "student_probability": adapter_scores[position],
                "student_margin": adapter_margins[position],
                "base_probability": base_scores[position],
                "base_margin": base_margins[position],
                "prompt_sha256": prompt_hashes[position],
            })
            handle.write(json.dumps(output, sort_keys=True) + "\n")

    manifest = {
        "method": args.method,
        "model": MODEL_ID,
        "adapter_dir": str(adapter_dir),
        "cache_dir": str(args.cache_dir.resolve()),
        "rows": len(rows),
        "dataset_units": len({row["dataset"] for row in rows}),
        "token_ids": {"0": token_ids[0], "1": token_ids[1]},
        "timing_seconds": {
            "adapter": adapter_seconds,
            "base": base_seconds,
        },
        "lora_fingerprint": {
            "exact_equal": sum(value == 0.0 for value in absolute_differences),
            "mean_absolute_probability_difference": (
                sum(absolute_differences) / len(absolute_differences)
            ),
            "maximum_absolute_probability_difference": max(absolute_differences),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    from experiments.kimi_imitation_audit.analyze import analyze

    report = analyze(output_dir / "scores.jsonl")
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print((output_dir / "summary.md").read_text(), end="")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.output_dir is None:
        args.output_dir = ROOT / "results/blackbox" / args.method
    return args


if __name__ == "__main__":
    run(parse_args())
