#!/usr/bin/env python3
"""Evaluate auxiliary label-token readouts at one fixed Q397 prompt boundary."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_ADAPTER = (
    ROOT
    / "results/blackbox/qwen9b_qwen397_tvg_soft_r16_lr5e5_ep2_v1/adapter"
)
DEFAULT_METHOD = "q397_auxiliary_readout_validation_v1"
PROMOTION_MACRO_DELTA = 0.0020
PROMOTION_INSTRUCTED_DELTA_FLOOR = -0.0010
PROMOTION_VARIED_DELTA_FLOOR = -0.0005


@dataclass(frozen=True)
class Readout:
    """One negative/positive token pair scored at the fixed digit prompt."""

    name: str
    negative_text: str
    positive_text: str


@dataclass(frozen=True)
class Blend:
    """A fixed probability mean over readouts from the same forward pass."""

    name: str
    members: tuple[str, ...]


READOUTS = (
    Readout("digits", "0", "1"),
    Readout("no_yes", "No", "Yes"),
    Readout("false_true", "False", "True"),
    Readout("a_b", "A", "B"),
    Readout("honest_deceptive", " honest", " deceptive"),
    Readout("truthful_misleading", " truthful", " misleading"),
)
BLENDS = (
    Blend("digits_no_yes_mean", ("digits", "no_yes")),
    Blend("digits_false_true_mean", ("digits", "false_true")),
    Blend("digits_a_b_mean", ("digits", "a_b")),
    Blend("digits_honest_deceptive_mean", ("digits", "honest_deceptive")),
    Blend("digits_truthful_misleading_mean", ("digits", "truthful_misleading")),
    Blend(
        "digits_semantic_mean",
        ("digits", "no_yes", "false_true", "honest_deceptive", "truthful_misleading"),
    ),
)


def resolve_token_ids(tokenizer: Any) -> tuple[dict[str, tuple[int, int]], list[int]]:
    """Resolve every readout to two distinct single tokens and their union."""
    pairs: dict[str, tuple[int, int]] = {}
    union: list[int] = []
    for readout in READOUTS:
        ids = []
        for text in (readout.negative_text, readout.positive_text):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(
                    f"{readout.name} label {text!r} tokenized as {encoded}; "
                    "expected one token"
                )
            token_id = int(encoded[0])
            ids.append(token_id)
            if token_id not in union:
                union.append(token_id)
        if ids[0] == ids[1]:
            raise ValueError(f"{readout.name} resolved to duplicate token ids")
        pairs[readout.name] = (ids[0], ids[1])
    return pairs, union


def logprob_value(value: Any) -> float:
    """Normalize vLLM float and Logprob wrappers."""
    return float(value.logprob if hasattr(value, "logprob") else value)


def requested_logprobs(output: Any, token_ids: list[int]) -> dict[int, float]:
    """Extract every explicitly requested next-token log probability."""
    if not output.outputs or not output.outputs[0].logprobs:
        raise ValueError("vLLM output is missing requested next-token logprobs")
    values = output.outputs[0].logprobs[0]
    result: dict[int, float] = {}
    for token_id in token_ids:
        value = values.get(token_id)
        if value is None:
            value = values.get(str(token_id))
        if value is None:
            raise ValueError(f"vLLM output is missing requested token id {token_id}")
        result[token_id] = logprob_value(value)
    return result


def sigmoid(value: float) -> float:
    """Return a numerically stable logistic score."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exponent = math.exp(max(value, -60.0))
    return exponent / (1.0 + exponent)


def scores_from_logprobs(
    rows: list[dict[int, float]],
    pairs: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Compute pair-normalized readout probabilities and frozen blends."""
    scores: dict[str, np.ndarray] = {}
    for readout in READOUTS:
        negative_id, positive_id = pairs[readout.name]
        scores[readout.name] = np.asarray(
            [
                sigmoid(values[positive_id] - values[negative_id])
                for values in rows
            ],
            dtype=np.float64,
        )
    for blend in BLENDS:
        scores[blend.name] = np.mean(
            np.stack([scores[member] for member in blend.members], axis=0),
            axis=0,
        )
    return scores


def scenario_summary(
    records: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[str, Any]:
    """Report competition macro metrics and per-dataset ranking."""
    from experiments.phoenix_verbalizer_sweep.run_ndif_base_verbalizers import (
        grouped_metrics,
        per_dataset_metrics,
    )

    rows = [
        {
            "dataset": record["dataset"],
            "index": record["index"],
            "label": record["label"],
            "score": float(score),
        }
        for record, score in zip(records, scores, strict=True)
    ]
    scenarios = {}
    for name in ("instructed-deception", "varied-deception"):
        subset = [row for row in rows if name in str(row["dataset"])]
        scenarios[name] = grouped_metrics(subset)
    return {
        "metrics": grouped_metrics(rows),
        "scenarios": scenarios,
        "per_dataset": per_dataset_metrics(rows),
        "score_stats": {
            "unique": int(np.unique(scores).size),
            "ties": int(len(scores) - np.unique(scores).size),
            "minimum": float(scores.min()),
            "mean": float(scores.mean()),
            "maximum": float(scores.max()),
        },
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    """Render a compact ranking table."""
    lines = [
        "| readout | kind | macro AUROC | Δ control | instructed | varied | unique |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['kind']} | {row['macro_auroc']:.6f} | "
            f"{row['delta_vs_digits']:+.6f} | {row['instructed_auroc']:.6f} | "
            f"{row['varied_auroc']:.6f} | {row['unique_scores']} |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the frozen validation sweep in one local vLLM model session."""
    from experiments.phoenix_verbalizer_sweep.run_ndif_base_verbalizers import (
        CONDITIONS,
        build_direct_prompt,
        load_validation_records,
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_dir = args.adapter_dir.resolve()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / filename).is_file():
            raise FileNotFoundError(adapter_dir / filename)
    records = load_validation_records(args.limit)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    pairs, union_ids = resolve_token_ids(tokenizer)

    # Every readout sees exactly this trained digit prompt and terminal boundary.
    digit_condition = next(
        condition for condition in CONDITIONS if condition.name == "digits_frozen"
    )
    prompts = [
        build_direct_prompt(record["messages"], tokenizer, digit_condition)
        for record in records
    ]
    prompt_hashes = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts
    ]

    llm = LLM(
        model=MODEL_ID,
        tokenizer=MODEL_ID,
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
        logprobs=len(union_ids),
        logprob_token_ids=union_ids,
        allowed_token_ids=union_ids,
    )
    request = LoRARequest(adapter_dir.parent.name, 1, adapter_dir.as_posix())

    started = time.perf_counter()
    adapter_outputs = llm.generate(prompts, sampling, lora_request=request)
    adapter_seconds = time.perf_counter() - started
    adapter_logprobs = [
        requested_logprobs(output, union_ids) for output in adapter_outputs
    ]
    scores = scores_from_logprobs(adapter_logprobs, pairs)

    # A matched base pass fingerprints that vLLM applied the LoRA.
    started = time.perf_counter()
    base_outputs = llm.generate(prompts, sampling)
    base_seconds = time.perf_counter() - started
    base_scores = scores_from_logprobs(
        [requested_logprobs(output, union_ids) for output in base_outputs],
        pairs,
    )["digits"]
    digit_delta = np.abs(scores["digits"].astype(np.float32) - base_scores.astype(np.float32))
    if np.all(digit_delta <= 1.0e-7):
        raise RuntimeError("LoRA effect fingerprint failed")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    score_rows = []
    for position, record in enumerate(records):
        score_rows.append({
            "dataset": record["dataset"],
            "index": record["index"],
            "label": record["label"],
            "prompt_sha256": prompt_hashes[position],
            "requested_logprobs": {
                str(token_id): adapter_logprobs[position][token_id]
                for token_id in union_ids
            },
            "scores": {
                name: float(values[position]) for name, values in scores.items()
            },
        })
    (output_dir / "scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in score_rows)
    )

    summaries = {
        name: scenario_summary(records, values) for name, values in scores.items()
    }
    digit = summaries["digits"]
    digit_macro = float(digit["metrics"]["auroc"])
    digit_instructed = float(
        digit["scenarios"]["instructed-deception"]["auroc"]
    )
    digit_varied = float(digit["scenarios"]["varied-deception"]["auroc"])
    readout_names = {readout.name for readout in READOUTS}
    rows = []
    for name, summary in summaries.items():
        rows.append({
            "name": name,
            "kind": "readout" if name in readout_names else "blend",
            "macro_auroc": float(summary["metrics"]["auroc"]),
            "delta_vs_digits": float(summary["metrics"]["auroc"]) - digit_macro,
            "instructed_auroc": float(
                summary["scenarios"]["instructed-deception"]["auroc"]
            ),
            "varied_auroc": float(
                summary["scenarios"]["varied-deception"]["auroc"]
            ),
            "unique_scores": int(summary["score_stats"]["unique"]),
        })
    rows.sort(key=lambda row: (-row["macro_auroc"], row["name"]))
    candidates = [row for row in rows if row["name"] != "digits"]
    best = candidates[0]
    best["passes_validation_gate"] = bool(
        best["delta_vs_digits"] >= PROMOTION_MACRO_DELTA
        and best["instructed_auroc"] - digit_instructed
        >= PROMOTION_INSTRUCTED_DELTA_FLOOR
        and best["varied_auroc"] - digit_varied >= PROMOTION_VARIED_DELTA_FLOOR
    )

    report = {
        "method": args.method,
        "model": MODEL_ID,
        "adapter_dir": str(adapter_dir),
        "rows": len(records),
        "fixed_prompt_condition": asdict(digit_condition),
        "readouts": [asdict(readout) for readout in READOUTS],
        "blends": [asdict(blend) for blend in BLENDS],
        "token_ids": {name: list(pair) for name, pair in pairs.items()},
        "requested_token_ids": union_ids,
        "timing": {
            "adapter_seconds": adapter_seconds,
            "base_fingerprint_seconds": base_seconds,
        },
        "lora_effect": {
            "mean_absolute_digit_difference": float(digit_delta.mean()),
            "max_absolute_digit_difference": float(digit_delta.max()),
            "exact_equal": int(np.equal(scores["digits"], base_scores).sum()),
        },
        "promotion_gate": {
            "macro_delta_min": PROMOTION_MACRO_DELTA,
            "instructed_delta_floor": PROMOTION_INSTRUCTED_DELTA_FLOOR,
            "varied_delta_floor": PROMOTION_VARIED_DELTA_FLOOR,
        },
        "best_candidate": best,
        "ranking": rows,
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "summary.md").write_text(markdown(rows))
    print(markdown(rows), end="")
    print(
        f"best={best['name']} delta={best['delta_vs_digits']:+.6f} "
        f"passes_validation_gate={best['passes_validation_gate']}",
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=DEFAULT_METHOD)
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


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
