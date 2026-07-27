#!/usr/bin/env python3
"""Evaluate continuous margins from the strong base-Qwen HP-KR specialist."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.evaluate_hpkr_rank1_specialist import (
    CATEGORY,
    render_prompts,
)
from experiments.liars_bench_distillation.evaluate_prompt_router import render
from experiments.liars_bench_distillation.evaluate_prompt_sweep import (
    load_prompt,
    split_records,
)
from experiments.liars_bench_distillation.evaluate_students import parse_prediction
from experiments.liars_bench_hpkr_prompts.evaluate import PROMPTS as HPKR_PROMPTS
from experiments.privileged_information_distillation.evaluate_student_sft import (
    binary_token_ids,
    continuous_margin_prompts,
    score_binary_prefixes,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import binary_metrics


MARGIN_ARMS = (
    "epistemic_direct",
    "epistemic_empty",
    "epistemic_reasoning",
)


def evaluated_rows(
    records: list[dict[str, Any]],
    scores: list[float],
) -> list[dict[str, Any]]:
    """Attach continuous scores to the frozen HP-KR records."""
    return [
        {
            **{
                field: record[field]
                for field in (
                    "dataset",
                    "index",
                    "category",
                    "source_model",
                    "label",
                )
            },
            "score": float(score),
        }
        for record, score in zip(records, scores, strict=True)
    ]


def grouped_metrics(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, float | None]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        name: binary_metrics(
            np.asarray([row["label"] for row in group], dtype=np.int64),
            np.asarray([row["score"] for row in group], dtype=np.float64),
            0.5,
        )
        for name, group in sorted(groups.items())
    }


def summarize(
    records: list[dict[str, Any]],
    scores: list[float],
    elapsed: float,
) -> dict[str, Any]:
    rows = evaluated_rows(records, scores)
    return {
        "rows": len(rows),
        "metrics": binary_metrics(
            np.asarray([row["label"] for row in rows], dtype=np.int64),
            np.asarray([row["score"] for row in rows], dtype=np.float64),
            0.5,
        ),
        "per_source_model": grouped_metrics(rows, "source_model"),
        "unique_scores": len({row["score"] for row in rows}),
        "score_seconds": elapsed,
    }


def select_margin_arm(
    summaries: dict[str, dict[str, Any]],
    *,
    minimum_auroc_gain: float,
) -> dict[str, Any]:
    """Select the best continuous epistemic arm only after a material gain."""
    baseline = float(summaries["phoenix_control"]["metrics"]["auroc"])
    comparisons = {}
    for name in MARGIN_ARMS:
        auroc = float(summaries[name]["metrics"]["auroc"])
        comparisons[name] = {
            "auroc": auroc,
            "gain_over_phoenix": auroc - baseline,
            "passes": auroc - baseline >= minimum_auroc_gain,
        }
    selected = max(
        (name for name in MARGIN_ARMS if comparisons[name]["passes"]),
        key=lambda name: (
            comparisons[name]["auroc"],
            -MARGIN_ARMS.index(name),
        ),
        default=None,
    )
    return {
        "metric": "hpkr_auroc",
        "baseline": "phoenix_control",
        "minimum_auroc_gain": minimum_auroc_gain,
        "selected": selected,
        "comparisons": comparisons,
    }


def load_selected(path: Path) -> str:
    selected = json.loads(path.read_text()).get("selected")
    if selected not in MARGIN_ARMS:
        raise ValueError(f"selection does not contain a valid margin arm: {selected!r}")
    return str(selected)


def generated_binary_scores(outputs: list[Any]) -> tuple[list[float], int]:
    """Represent parsed generated decisions as deliberately near-binary scores."""
    scores = []
    parse_errors = 0
    for output in outputs:
        text = output.outputs[0].text if output.outputs else ""
        prediction = parse_prediction(text)
        if prediction is None:
            parse_errors += 1
            prediction = 0
        scores.append(0.500001 if prediction == 1 else 0.499999)
    return scores, parse_errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-artifact", type=Path, required=True)
    parser.add_argument("--phoenix-adapter", type=Path, required=True)
    parser.add_argument("--phoenix-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("development", "confirmation"),
        required=True,
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260725)
    parser.add_argument("--expected-rows", type=int, default=100)
    parser.add_argument("--minimum-auroc-gain", type=float, default=0.05)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument("--max-generation-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if args.split == "confirmation" and args.selection is None:
        raise ValueError("--selection is required for confirmation")
    if args.split == "development" and args.selection is not None:
        raise ValueError("--selection is only valid for confirmation")

    all_records = [
        json.loads(line)
        for line in args.eval_artifact.read_text().splitlines()
        if line.strip()
    ]
    records = [
        row
        for row in split_records(all_records, args.split, seed=args.split_seed)
        if row["category"] == CATEGORY
    ]
    if len(records) != args.expected_rows:
        raise ValueError(
            f"{args.split} contains {len(records)} HP-KR rows, "
            f"expected {args.expected_rows}"
        )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=16,
    )
    binary_ids = binary_token_ids(tokenizer)
    binary_sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(binary_ids),
        logprob_token_ids=binary_ids,
        allowed_token_ids=binary_ids,
    )
    generation_sampling = SamplingParams(
        max_tokens=args.max_generation_tokens,
        temperature=0.0,
    )
    phoenix_request = LoRARequest(
        "phoenix-control",
        1,
        args.phoenix_adapter.resolve().as_posix(),
    )

    phoenix_prompts = [
        prompt + "Prediction:"
        for prompt in render(
            tokenizer,
            records,
            load_prompt(args.phoenix_config.resolve()),
            max_prompt_chars=args.max_prompt_chars,
            context_truncation="head_tail",
        )
    ]
    phoenix_scores, phoenix_missing, phoenix_elapsed = score_binary_prefixes(
        llm,
        phoenix_prompts,
        binary_sampling,
        phoenix_request,
        binary_ids,
    )

    epistemic_prompts = render_prompts(
        tokenizer,
        records,
        HPKR_PROMPTS["knowledge_report_type"],
        max_prompt_chars=args.max_prompt_chars,
    )
    started = time.time()
    generated = llm.generate(
        epistemic_prompts,
        generation_sampling,
        lora_request=None,
    )
    generation_elapsed = time.time() - started
    generations = [
        output.outputs[0].text if output.outputs else ""
        for output in generated
    ]
    generated_scores, parse_errors = generated_binary_scores(generated)

    margin_prompts = [
        continuous_margin_prompts(prompt, generation)
        for prompt, generation in zip(
            epistemic_prompts,
            generations,
            strict=True,
        )
    ]
    all_arm_scores: dict[str, list[float]] = {}
    missing: dict[str, int] = {"phoenix_control": phoenix_missing}
    timing: dict[str, float] = {
        "phoenix_control": phoenix_elapsed,
        "epistemic_generation": generation_elapsed,
    }
    for name, prompt_key in (
        ("epistemic_direct", "direct"),
        ("epistemic_empty", "empty"),
        ("epistemic_reasoning", "reasoning"),
    ):
        values, arm_missing, elapsed = score_binary_prefixes(
            llm,
            [row[prompt_key] for row in margin_prompts],
            binary_sampling,
            None,
            binary_ids,
        )
        all_arm_scores[name] = values
        missing[name] = arm_missing
        timing[name] = elapsed

    summaries = {
        "phoenix_control": summarize(
            records,
            phoenix_scores,
            phoenix_elapsed,
        ),
        "epistemic_generated_binary": summarize(
            records,
            generated_scores,
            generation_elapsed,
        ),
    }
    for name, scores in all_arm_scores.items():
        summaries[name] = summarize(records, scores, timing[name])

    if args.split == "development":
        emitted_arms = MARGIN_ARMS
        selection = select_margin_arm(
            summaries,
            minimum_auroc_gain=args.minimum_auroc_gain,
        )
    else:
        selected = load_selected(args.selection.resolve())
        emitted_arms = (selected,)
        selection = {
            "metric": "hpkr_auroc",
            "frozen_selection": selected,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_sets = {
        "phoenix_control": phoenix_scores,
        "epistemic_generated_binary": generated_scores,
        **{name: all_arm_scores[name] for name in emitted_arms},
    }
    for name, scores in score_sets.items():
        with (args.output_dir / f"{name}.jsonl").open("w") as handle:
            for row in evaluated_rows(records, scores):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.split == "development":
        (args.output_dir / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n"
        )
    result = {
        "split": args.split,
        "split_seed": args.split_seed,
        "rows": len(records),
        "prompt": "knowledge_report_type",
        "parse_errors": parse_errors,
        "margin_missing": missing,
        "timing": timing,
        "conditions": summaries,
        "selection": selection,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
