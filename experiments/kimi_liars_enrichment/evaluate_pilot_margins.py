#!/usr/bin/env python3
"""Evaluate adapter direct margins on the frozen 800-row Liars pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kimi_liars_enrichment.prompts import CATEGORY_SPECS
from experiments.privileged_information_distillation.core import build_student_prompt
from experiments.privileged_information_distillation.evaluate_student_sft import (
    binary_token_ids,
    metrics_for_score,
    score_binary_prefixes,
)


def parse_adapter(value: str) -> tuple[str, Path]:
    """Parse a stable condition name and adapter path."""
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"invalid adapter {value!r}; expected NAME=PATH")
    return name, Path(path).resolve()


def render_prompts(records: list[dict[str, Any]], tokenizer: Any) -> list[str]:
    """Render each row with the Phoenix 8 route used for student training."""
    prompts = []
    for record in records:
        category = str(record["category"])
        spec = CATEGORY_SPECS[category]
        prompt = build_student_prompt(
            record["messages"],
            str(spec["student_prompt"]),
            int(spec["student_max_chars"]),
            str(spec["student_truncation"]),
        )
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(chat + "Prediction:")
    return prompts


def grouped_auroc(
    frame: pd.DataFrame, fields: list[str]
) -> dict[str, float | None]:
    """Report AUROC for groups that contain both labels."""
    result: dict[str, float | None] = {}
    group_key: str | list[str] = fields[0] if len(fields) == 1 else fields
    for key, group in frame.groupby(group_key, sort=True):
        name = key if isinstance(key, str) else "::".join(map(str, key))
        result[str(name)] = (
            float(roc_auc_score(group["label"], group["score"]))
            if group["label"].nunique() == 2
            else None
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", action="append", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    records = [
        json.loads(line)
        for line in args.eval_artifact.read_text().splitlines()
        if line.strip()
    ]
    if len(records) != 800:
        raise ValueError(f"expected the frozen 800-row pilot, found {len(records)}")
    adapters = [parse_adapter(value) for value in args.adapter]
    if len({name for name, _ in adapters}) != len(adapters):
        raise ValueError("adapter names must be unique")

    tokenizer = AutoTokenizer.from_pretrained(adapters[0][1])
    prompts = render_prompts(records, tokenizer)
    token_ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(token_ids),
        logprob_token_ids=token_ids,
        allowed_token_ids=token_ids,
    )
    llm = LLM(
        model=args.model,
        tokenizer=adapters[0][1].as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=4096,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"rows": len(records), "conditions": {}}
    for lora_id, (name, path) in enumerate(adapters, start=1):
        scores, missing, elapsed = score_binary_prefixes(
            llm,
            prompts,
            sampling,
            LoRARequest(name, lora_id, path.as_posix()),
            token_ids,
        )
        frame = pd.DataFrame(
            {
                "dataset": [
                    f"liars-bench/{record['category']}" for record in records
                ],
                "index": [record["index"] for record in records],
                "category": [record["category"] for record in records],
                "source_model": [record["source_model"] for record in records],
                "label": [int(record["label"]) for record in records],
                "score": scores,
            }
        )
        frame.to_json(
            args.output_dir / f"{name}.jsonl",
            orient="records",
            lines=True,
            double_precision=15,
        )
        metrics = metrics_for_score(frame, "score")
        condition = {
            "metrics": metrics,
            "per_category_auroc": grouped_auroc(frame, ["category"]),
            "per_category_source_auroc": grouped_auroc(
                frame, ["category", "source_model"]
            ),
            "unique_scores": int(frame["score"].nunique()),
            "missing_label_logits": missing,
            "score_seconds": elapsed,
        }
        summary["conditions"][name] = condition
        print(
            f"{name} macro_auroc={metrics['all']['auroc']:.6f} "
            f"unique={condition['unique_scores']} missing={missing} "
            f"time={elapsed:.1f}s",
            flush=True,
        )
    (args.output_dir / "result.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
