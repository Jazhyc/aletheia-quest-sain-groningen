#!/usr/bin/env python3
"""Use GPT-OSS to audit evidence and generate queries only after abstention."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.selective_hop import (
    bounded_candidates,
    build_audit_prompt,
    claim_key,
    parse_audit_completion,
)
from experiments.privileged_information_distillation.core import extract_harmony_final


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: row[field]
        for field in (
            "dataset", "index", "claim_index", "label", "quote", "proposition",
            "question", "teacher_assessment",
        )
        if field in row
    }


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-kind", choices=("verification", "retrieval"), required=True)
    parser.add_argument("--previous-audit", type=Path)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--max-candidate-chars", type=int, default=1200)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.input_kind == "verification":
        rows = [row for row in rows if row.get("condition") == "real"]
        source_field = "evidence"
    else:
        source_field = "passages"
    if args.previous_audit:
        previous = {
            claim_key(row): row for row in read_jsonl(args.previous_audit)
        }
        rows = [
            row for row in rows
            if previous.get(claim_key(row), {}).get("decision") == "ABSTAIN"
        ]
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    metadata = []
    prompts = []
    for row in rows:
        candidates = bounded_candidates(
            row,
            source_field=source_field,
            limit=args.max_candidates,
            max_chars=args.max_candidate_chars,
            offset=args.candidate_offset,
        )
        raw_prompt = build_audit_prompt(str(row["proposition"]), candidates)
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ))
        metadata.append((row, candidates, raw_prompt))

    model = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        max_num_seqs=256,
    )
    started = time.time()
    outputs = model.generate(
        prompts,
        SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
    )
    elapsed = time.time() - started
    records = []
    for (row, candidates, raw_prompt), output in zip(metadata, outputs, strict=True):
        raw = output.outputs[0].text if output.outputs else ""
        final = extract_harmony_final(raw)
        parsed = parse_audit_completion(
            final,
            candidates=candidates,
            proposition=str(row["proposition"]),
        )
        records.append({
            **identity(row),
            "stage": "second" if args.previous_audit else "initial",
            "candidates": candidates,
            "prompt": raw_prompt,
            "raw_completion": raw,
            "harmony_final": final,
            **parsed,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n"
    )
    summary = {
        "rows": len(records),
        "decisions": Counter(row["decision"] for row in records),
        "parse_errors": sum(not row["parse_valid"] for row in records),
        "score_seconds": elapsed,
        "rows_per_second": len(records) / elapsed if elapsed else 0.0,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
