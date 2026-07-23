#!/usr/bin/env python3
"""Run empty, real, and shuffled Wikidata evidence through one frozen student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.evaluate_student_sft import (
    EMPTY_REASONING_PREFIX,
    STRICT_RE,
    binary_token_ids,
    evaluate_adapter,
    load_records,
    metrics_for_score,
)


EMPTY_REFERENCE = "- No retrieved reference material."
CONDITIONS = ("empty", "real", "shuffled")
SCORE_COLUMNS = ("score", "empty_margin_score", "reasoning_margin_score")
OUTPUT_COLUMNS = (
    "prediction", "score", "parse_error", "format_valid", "generation",
    "empty_margin_score", "reasoning_margin_score",
)


def format_passages(passages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- {passage.get('title', '')}: {passage.get('text', '')}"
        for passage in passages
        if passage.get("text")
    )


def load_condition_references(
    cache_path: Path,
) -> tuple[dict[str, dict[tuple[str, Any], str]], set[tuple[str, Any]]]:
    references = {condition: {} for condition in CONDITIONS}
    varied_keys = set()
    for line in cache_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["dataset"], row["index"])
        varied_keys.add(key)
        references["real"][key] = format_passages(row.get("real_passages", []))
        references["shuffled"][key] = format_passages(row.get("shuffled_passages", []))
    return references, varied_keys


def active_evidence_keys(cache_path: Path) -> set[tuple[str, Any]]:
    """Return rows whose real condition actually differs from empty."""
    active = set()
    for line in cache_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if format_passages(row.get("real_passages", [])):
            active.add((row["dataset"], row["index"]))
    return active


def reuse_empty_for_inactive_rows(
    evaluated: pd.DataFrame, active_keys: set[tuple[str, Any]]
) -> pd.DataFrame:
    """Remove condition noise where sparse retrieval supplied identical prompts."""
    result = evaluated.copy()
    empty = result[result["condition"] == "empty"].set_index(["dataset", "index"])
    columns = [column for column in OUTPUT_COLUMNS if column in result.columns]
    for row_index, row in result[result["condition"] != "empty"].iterrows():
        key = (row["dataset"], row["index"])
        if key in active_keys:
            continue
        for column in columns:
            result.at[row_index, column] = empty.at[key, column]
    return result


def paired_changes(
    evaluated: pd.DataFrame,
    condition: str,
    baseline: str = "empty",
) -> dict[str, int]:
    subset = evaluated[evaluated["dataset"].str.contains("varied-deception")]
    pivot = subset.pivot(index=["dataset", "index", "label"], columns="condition", values="prediction")
    pivot = pivot.dropna(subset=[baseline, condition])
    labels = pivot.index.get_level_values("label")
    before = pivot[baseline].astype(int)
    after = pivot[condition].astype(int)
    return {
        "deceptive_recoveries_fn_to_tp": int(((labels == 1) & (before == 0) & (after == 1)).sum()),
        "deceptive_regressions_tp_to_fn": int(((labels == 1) & (before == 1) & (after == 0)).sum()),
        "honest_harms_tn_to_fp": int(((labels == 0) & (before == 0) & (after == 1)).sum()),
        "honest_recoveries_fp_to_tn": int(((labels == 0) & (before == 1) & (after == 0)).sum()),
        "paired_rows": int(len(pivot)),
    }


def paired_score_deltas(
    evaluated: pd.DataFrame,
    condition: str,
    score_column: str,
    baseline: str = "empty",
) -> dict[str, float]:
    subset = evaluated[evaluated["dataset"].str.contains("varied-deception")]
    pivot = subset.pivot(index=["dataset", "index", "label"], columns="condition", values=score_column)
    delta = pivot[condition] - pivot[baseline]
    labels = pivot.index.get_level_values("label")
    return {
        "all": float(delta.mean()),
        "honest": float(delta[labels == 0].mean()),
        "deceptive": float(delta[labels == 1].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    adapter_dir = args.adapter_dir.resolve()
    config = yaml.safe_load((adapter_dir.parent / "config.yaml").read_text())
    references, varied_keys = load_condition_references(args.cache.resolve())
    active_keys = active_evidence_keys(args.cache.resolve())

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_records = load_records(
        args.split,
        args.splits_dir.resolve(),
        config,
        tokenizer,
    )
    all_keys = {
        (row.dataset, row.index)
        for row in base_records.itertuples()
    }
    expected_varied_keys = {
        key for key in all_keys if "varied-deception" in key[0]
    }
    if varied_keys != expected_varied_keys:
        missing = len(expected_varied_keys - varied_keys)
        extra = len(varied_keys - expected_varied_keys)
        raise RuntimeError(
            f"retrieval cache does not match the split: missing={missing}, extra={extra}"
        )
    frames = []
    for condition in CONDITIONS:
        condition_references = references[condition]
        # All three conditions use an explicit reference block. Instructed rows
        # and the empty condition receive the same sentinel text.
        records = load_records(
            args.split,
            args.splits_dir.resolve(),
            config,
            tokenizer,
            references={
                key: condition_references.get(key) or EMPTY_REFERENCE
                for key in all_keys
            },
        )
        records["condition"] = condition
        frames.append(records)
    records = pd.concat(frames, ignore_index=True)
    print(
        f"loaded {len(records)} prompts: "
        + ", ".join(f"{name}={(records['condition'] == name).sum()}" for name in CONDITIONS),
        flush=True,
    )

    llm = LLM(
        model=config["student"]["model"],
        tokenizer=adapter_dir.as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=int(config["student"]["lora"]["r"]),
        max_model_len=args.max_model_len,
    )
    generation_sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    binary_ids = binary_token_ids(tokenizer)
    margin_sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(binary_ids),
        logprob_token_ids=binary_ids,
        allowed_token_ids=binary_ids,
    )
    evaluated, timing = evaluate_adapter(
        llm,
        generation_sampling,
        records,
        adapter_dir,
        1,
        strict_re=STRICT_RE,
        margin_sampling=margin_sampling,
        binary_ids=binary_ids,
        empty_reasoning_prefix=EMPTY_REASONING_PREFIX,
    )
    evaluated = reuse_empty_for_inactive_rows(evaluated, active_keys)

    condition_metrics = {}
    for condition in CONDITIONS:
        subset = evaluated[evaluated["condition"] == condition]
        condition_metrics[condition] = {
            score_column: metrics_for_score(subset, score_column)
            for score_column in SCORE_COLUMNS
        }
    comparisons = {}
    for condition in ("real", "shuffled"):
        comparisons[f"{condition}_vs_empty"] = {
            "transitions": paired_changes(evaluated, condition),
            "score_deltas": {
                score_column: paired_score_deltas(evaluated, condition, score_column)
                for score_column in SCORE_COLUMNS
            },
        }
    comparisons["real_vs_shuffled"] = {
        "transitions": paired_changes(evaluated, "real", baseline="shuffled"),
        "score_deltas": {
            score_column: paired_score_deltas(
                evaluated, "real", score_column, baseline="shuffled"
            )
            for score_column in SCORE_COLUMNS
        },
    }
    result = {
        "method": config["method"],
        "split": args.split,
        "rows_per_condition": int(len(evaluated) / len(CONDITIONS)),
        "varied_rows_per_condition": int(
            len(evaluated[evaluated["dataset"].str.contains("varied-deception")])
            / len(CONDITIONS)
        ),
        "active_varied_rows": len(active_keys),
        "inactive_outputs_reused": True,
        "conditions": condition_metrics,
        "comparisons": comparisons,
        "parse_errors": {
            condition: int(
                evaluated.loc[evaluated["condition"] == condition, "parse_error"].sum()
            )
            for condition in CONDITIONS
        },
        "timing": timing,
        "cache": args.cache.resolve().as_posix(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluated.to_json(args.output_dir / "generations.jsonl", orient="records", lines=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
