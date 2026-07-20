#!/usr/bin/env python3
"""Evaluate evidence-trained readers under paired real/empty/shuffled controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.evaluate_student_sft import (
    STRICT_RE,
    evaluate_adapter,
    load_records,
    load_retrieval_cache,
    metrics_for_score,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import binary_metrics


def load_correctness_records(path: Path, tokenizer: Any) -> pd.DataFrame:
    """Render an independent correctness cache for matched adapter evaluation."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(record["student_prompt"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rows.append({
            "dataset": str(record["dataset"]),
            "index": record["index"],
            "label": int(record["label"]),
            "condition": str(record["condition"]),
            "question_group": str(record["question_group"]),
            "prompt": prompt,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"no correctness records in {path}")
    return frame


def correctness_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Report balanced factual accuracy and condition-specific firing rates."""
    labels = frame["label"].to_numpy()
    scores = frame["score"].to_numpy()
    result: dict[str, Any] = {
        "all": binary_metrics(labels, scores, 0.5),
        "conditions": {},
    }
    for condition, group in frame.groupby("condition", sort=True):
        predictions = group["score"].to_numpy() >= 0.5
        result["conditions"][condition] = {
            "rows": int(len(group)),
            "accuracy": float((predictions == group["label"].to_numpy()).mean()),
            "positive_rate": float(predictions.mean()),
            "parse_errors": int(group["parse_error"].sum()),
        }
    negative_controls = frame[frame["condition"] != "refute"]
    result["negative_control_fpr"] = float(
        (negative_controls["score"].to_numpy() >= 0.5).mean()
    )
    refutations = frame[frame["condition"] == "refute"]
    result["refutation_recall"] = float(
        (refutations["score"].to_numpy() >= 0.5).mean()
    )
    return result


def paired_changes(
    candidate: pd.DataFrame,
    control: pd.DataFrame,
    *,
    scenario: str = "varied-deception",
) -> dict[str, int]:
    """Count prediction changes and their correctness effect on matched rows."""
    columns = ["dataset", "index", "label", "prediction"]
    left = candidate[columns].rename(columns={"prediction": "candidate"})
    right = control[columns].rename(columns={"prediction": "control"})
    right = right.drop(columns="label")
    merged = left.merge(right, on=["dataset", "index"], validate="one_to_one")
    if scenario:
        merged = merged[merged["dataset"].str.contains(scenario)]
    candidate_values = merged["candidate"].fillna(-1).astype(int)
    control_values = merged["control"].fillna(-1).astype(int)
    candidate_correct = candidate_values == merged["label"]
    control_correct = control_values == merged["label"]
    changed = candidate_values != control_values
    return {
        "rows": int(len(merged)),
        "changed_predictions": int(changed.sum()),
        "fixed": int((candidate_correct & ~control_correct).sum()),
        "broken": int((~candidate_correct & control_correct).sum()),
        "net_correct": int(candidate_correct.sum() - control_correct.sum()),
    }


def condition_references(
    condition: str,
    cache: Path,
) -> dict[tuple[str, Any], str]:
    if condition == "empty":
        return {}
    field = {"real": "real_passages", "shuffled": "shuffled_passages"}[condition]
    return load_retrieval_cache(cache, passage_field=field)


def changed_prompt_mask(records: pd.DataFrame, empty_records: pd.DataFrame) -> pd.Series:
    """Identify rows whose evidence condition actually changes the rendered prompt."""
    keys = ["dataset", "index"]
    if not records[keys].equals(empty_records[keys]):
        raise ValueError("evidence conditions do not contain the same ordered rows")
    return records["prompt"] != empty_records["prompt"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", action="append", type=Path, required=True)
    parser.add_argument("--retrieval-cache", type=Path, required=True)
    parser.add_argument(
        "--correctness-validation-cache",
        type=Path,
        help="optional entity-disjoint correctness cache evaluated in the same session",
    )
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--condition", action="append", choices=["empty", "real", "shuffled"])
    parser.add_argument("--run-name", default="validation_evidence_ablation")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    adapter_dirs = [path.resolve() for path in args.adapter_dir]
    configs = [yaml.safe_load((path.parent / "config.yaml").read_text()) for path in adapter_dirs]
    models = {config["student"]["model"] for config in configs}
    if len(models) != 1:
        raise SystemExit(f"adapters use different base models: {sorted(models)}")
    conditions = list(dict.fromkeys(args.condition or ["empty", "real", "shuffled"]))
    if "empty" in conditions:
        conditions.remove("empty")
        conditions.insert(0, "empty")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizers = [AutoTokenizer.from_pretrained(path) for path in adapter_dirs]
    correctness_records = None
    if args.correctness_validation_cache is not None:
        correctness_records = load_correctness_records(
            args.correctness_validation_cache.resolve(), tokenizers[0]
        )
    cache = args.retrieval_cache.resolve()
    reference_sets = {
        condition: condition_references(condition, cache)
        for condition in conditions
    }
    records: dict[tuple[int, str], pd.DataFrame] = {}
    for adapter_number, (config, tokenizer) in enumerate(zip(configs, tokenizers, strict=True)):
        for condition in conditions:
            records[(adapter_number, condition)] = load_records(
                args.split,
                args.splits_dir.resolve(),
                config,
                tokenizer,
                references=reference_sets[condition],
                append_empty_reference=True,
            )

    first = configs[0]
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
    all_results: dict[str, Any] = {}
    all_frames: dict[tuple[str, str], pd.DataFrame] = {}

    for adapter_number, (adapter_dir, config) in enumerate(
        zip(adapter_dirs, configs, strict=True)
    ):
        method = str(config["method"])
        method_result: dict[str, Any] = {
            "conditions": {},
            "paired": {},
            "correctness": None,
        }
        for condition in conditions:
            condition_records = records[(adapter_number, condition)]
            reused_empty_rows = 0
            if condition != "empty" and (method, "empty") in all_frames:
                empty_records = records[(adapter_number, "empty")]
                changed = changed_prompt_mask(condition_records, empty_records)
                frame = all_frames[(method, "empty")].copy()
                if changed.any():
                    partial, timing = evaluate_adapter(
                        llm,
                        sampling,
                        condition_records[changed],
                        adapter_dir,
                        adapter_number + 1,
                        strict_re=STRICT_RE,
                    )
                    frame.loc[partial.index, partial.columns] = partial
                else:
                    timing = {"generation_seconds": 0.0}
                reused_empty_rows = int((~changed).sum())
            else:
                frame, timing = evaluate_adapter(
                    llm,
                    sampling,
                    condition_records,
                    adapter_dir,
                    adapter_number + 1,
                    strict_re=STRICT_RE,
                )
            all_frames[(method, condition)] = frame
            metrics = metrics_for_score(frame, "score")
            output_dir = adapter_dir.parent / args.run_name / condition
            output_dir.mkdir(parents=True, exist_ok=True)
            frame.to_json(output_dir / "generations.jsonl", orient="records", lines=True)
            per_dataset = {
                dataset: binary_metrics(
                    group["label"].to_numpy(), group["score"].to_numpy(), 0.5
                )
                for dataset, group in frame.groupby("dataset", sort=True)
            }
            result = {
                "method": method,
                "split": args.split,
                "condition": condition,
                "metrics": metrics,
                "per_dataset": per_dataset,
                "parse_errors": int(frame["parse_error"].sum()),
                "format_valid": int(frame["format_valid"].sum()),
                "rows": int(len(frame)),
                "generated_rows": int(len(frame) - reused_empty_rows),
                "reused_empty_rows": reused_empty_rows,
                "score_seconds": float(timing["generation_seconds"]),
                "retrieval_cache": cache.as_posix(),
                "passage_field": {
                    "empty": None,
                    "real": "real_passages",
                    "shuffled": "shuffled_passages",
                }[condition],
            }
            (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            method_result["conditions"][condition] = result
            print(
                f"{method} {condition}: {metrics['all']} varied={metrics.get('varied')} "
                f"parse_errors={result['parse_errors']}",
                flush=True,
            )
        if "real" in conditions:
            for control in ("empty", "shuffled"):
                if control in conditions:
                    method_result["paired"][f"real_vs_{control}"] = paired_changes(
                        all_frames[(method, "real")], all_frames[(method, control)]
                    )
        if correctness_records is not None:
            correctness_frame, correctness_timing = evaluate_adapter(
                llm,
                sampling,
                correctness_records,
                adapter_dir,
                adapter_number + 1,
                strict_re=STRICT_RE,
            )
            correctness_result = {
                "metrics": correctness_metrics(correctness_frame),
                "rows": int(len(correctness_frame)),
                "score_seconds": float(correctness_timing["generation_seconds"]),
                "cache": args.correctness_validation_cache.resolve().as_posix(),
            }
            correctness_output = adapter_dir.parent / args.run_name / "correctness"
            correctness_output.mkdir(parents=True, exist_ok=True)
            correctness_frame.to_json(
                correctness_output / "generations.jsonl",
                orient="records",
                lines=True,
            )
            (correctness_output / "result.json").write_text(
                json.dumps(correctness_result, indent=2) + "\n"
            )
            method_result["correctness"] = correctness_result
            print(
                f"{method} correctness: {correctness_result['metrics']}",
                flush=True,
            )
        all_results[method] = method_result

    summary = {
        "split": args.split,
        "conditions": conditions,
        "generated_at_unix": time.time(),
        "adapters": all_results,
    }
    summary_path = adapter_dirs[0].parent / args.run_name / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({method: row["paired"] for method, row in all_results.items()}, indent=2))
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
