#!/usr/bin/env python3
"""Screen disjoint honest Alpaca rows for Phoenix false positives."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.evaluate_students import parse_prediction
from experiments.liars_bench_distillation.prepare_teacher_data import load_category
from experiments.privileged_information_distillation.core import build_student_prompt


def select_per_model(
    records: list[dict[str, Any]], maximum_per_model: int, seed: int
) -> list[dict[str, Any]]:
    """Select a stable bounded set of hard negatives from each source family."""
    if maximum_per_model <= 0:
        raise ValueError("maximum_per_model must be positive")
    groups: dict[str, list[tuple[bytes, dict[str, Any]]]] = defaultdict(list)
    for record in records:
        digest = hashlib.sha256(
            f"{seed}\0{record['index']}".encode("utf-8")
        ).digest()
        groups[str(record["source_model"])].append((digest, record))
    selected = []
    for model in sorted(groups):
        selected.extend(
            record for _, record in sorted(groups[model])[:maximum_per_model]
        )
    return sorted(selected, key=lambda record: str(record["index"]))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--signature-artifact", type=Path, required=True)
    parser.add_argument("--screen-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--eval-artifact", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-per-model", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max-prompt-chars", type=int, default=3000)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--liars-root", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    frame = load_category("alpaca", args.liars_root)
    by_id = {str(row.sample_id): row for row in frame.itertuples(index=False)}
    signature_rows = [
        json.loads(line) for line in args.signature_artifact.read_text().splitlines()
        if line.strip()
    ]
    frozen_ids = {
        str(row["sample_id"]) for row in signature_rows
        if row.get("category") == "alpaca"
    }
    if len(frozen_ids) != 400:
        raise RuntimeError(f"expected 400 frozen Alpaca ids, found {len(frozen_ids)}")
    missing = frozen_ids - by_id.keys()
    if missing:
        raise RuntimeError(f"frozen Alpaca ids missing from public data: {len(missing)}")

    evaluation = [
        {
            "dataset": "liars-bench/alpaca",
            "index": sample_id,
            "category": "alpaca",
            "source_model": str(by_id[sample_id].source_model),
            "label": 0,
            "messages": by_id[sample_id].messages,
        }
        for sample_id in sorted(frozen_ids)
    ]
    write_jsonl(args.eval_artifact, evaluation)

    config = OmegaConf.load(ROOT / "configs/privileged_information_distillation.yaml")
    prompt_template = str(config.student.prompt)
    screen_records = [
        {
            "dataset": "liars-bench/alpaca-hallucination-control",
            "index": str(row.sample_id),
            "category": "alpaca",
            "source_model": str(row.source_model),
            "label": 0,
            "messages": row.messages,
            "student_prompt": build_student_prompt(
                row.messages, prompt_template, args.max_prompt_chars, "tail"
            ),
        }
        for row in frame.itertuples(index=False)
        if str(row.sample_id) not in frozen_ids
    ]
    if len(screen_records) != 7600:
        raise RuntimeError(f"expected 7600 disjoint screen rows, found {len(screen_records)}")

    adapter = args.adapter.resolve()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["student_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in screen_records
    ]
    raw_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered]
    token_lengths = [len(ids) for ids in raw_ids]
    prompts = [
        TokensPrompt(prompt_token_ids=ids[-args.max_input_tokens:]) for ids in raw_ids
    ]
    llm = LLM(
        model=args.model,
        tokenizer=adapter.as_posix(),
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enable_lora=True,
        max_lora_rank=16,
    )
    started = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
        lora_request=LoRARequest("phoenix", 1, adapter.as_posix()),
    )
    elapsed = time.perf_counter() - started
    screened = []
    for row, output in zip(screen_records, outputs, strict=True):
        generation = output.outputs[0].text if output.outputs else ""
        parsed = parse_prediction(generation)
        screened.append({
            **row,
            "prediction": 0 if parsed is None else parsed,
            "parse_error": parsed is None,
            "generation": generation,
        })
    positives = [row for row in screened if row["prediction"] == 1]
    candidates = select_per_model(positives, args.max_per_model, args.seed)
    write_jsonl(args.screen_artifact, screened)
    write_jsonl(args.candidate_artifact, candidates)
    result = {
        "rows": len(screened),
        "frozen_eval_rows": len(evaluation),
        "positive_rows": len(positives),
        "positive_rate": len(positives) / len(screened),
        "parse_errors": sum(row["parse_error"] for row in screened),
        "positive_per_source_model": dict(sorted(Counter(
            row["source_model"] for row in positives
        ).items())),
        "selected_rows": len(candidates),
        "selected_per_source_model": dict(sorted(Counter(
            row["source_model"] for row in candidates
        ).items())),
        "max_prompt_tokens_before_truncation": max(token_lengths),
        "token_truncated_rows": sum(
            length > args.max_input_tokens for length in token_lengths
        ),
        "score_seconds": elapsed,
    }
    result_path = args.candidate_artifact.with_name("screen_result.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if len(candidates) < 24:
        raise RuntimeError(
            f"only {len(candidates)} hard negatives selected; require at least 24"
        )


if __name__ == "__main__":
    main()
