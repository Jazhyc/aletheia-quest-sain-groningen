#!/usr/bin/env python3
"""Paraphrase mechanically grounded retrieval examples with local GPT-OSS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any

from experiments.privileged_information_distillation.core import extract_harmony_final
from experiments.wikidata_rag.claim_retrieval import normalize


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompt_for(row: dict[str, Any], count: int) -> str:
    truth_note = (
        "The answer is intentionally false. Do not correct it."
        if row["false_answer"] else "The answer is intentionally true."
    )
    return f"""Paraphrase this factual question and answer in {count} substantially
different natural styles for retrieval training. {truth_note}

Preserve the exact entity string `{row['anchor_subject']}` somewhere in
each question and preserve the exact answer-value string `{row['answer_value']}`
somewhere in each answer. Do not add factual details, explanations, hedging, or
corrections. Keep each question and answer to one sentence. Return only a JSON
object with key `pairs`, whose value is an array of objects with string keys
`question` and `answer`.

Canonical question: {row['question']}
Canonical answer: {row['answer_full']}
"""


def parse_pairs(completion: str) -> list[dict[str, str]]:
    final = extract_harmony_final(completion)
    starts = [match.start() for match in re.finditer(r"\{", final)]
    for start in starts:
        for end in range(len(final), start, -1):
            if final[end - 1] != "}":
                continue
            try:
                value = json.loads(final[start:end])
            except json.JSONDecodeError:
                continue
            pairs = value.get("pairs") if isinstance(value, dict) else None
            if isinstance(pairs, list) and all(
                isinstance(item, dict) and isinstance(item.get("question"), str)
                and isinstance(item.get("answer"), str) for item in pairs
            ):
                return pairs
    return []


def valid_pair(pair: dict[str, str], row: dict[str, Any]) -> bool:
    subject = normalize(row["anchor_subject"])
    value = normalize(row["answer_value"])
    return subject in normalize(pair["question"]) and value in normalize(pair["answer"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--paraphrases", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load(args.input)[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_for(row, args.paraphrases)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) for row in rows]
    llm = LLM(
        model=args.model, dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=0.9, max_model_len=4096,
    )
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.7, seed=42)
    started = time.time()
    outputs = []
    for start in range(0, len(prompts), args.batch_size):
        outputs.extend(llm.generate(prompts[start:start + args.batch_size], sampling))
    elapsed = time.time() - started

    augmented = []
    failures = 0
    for row, output in zip(rows, outputs, strict=True):
        completion = output.outputs[0].text if output.outputs else ""
        pairs = [pair for pair in parse_pairs(completion) if valid_pair(pair, row)]
        if not pairs:
            failures += 1
        for variant, pair in enumerate(pairs[:args.paraphrases]):
            labels = [
                {
                    **label,
                    "claim_quote": pair["answer"] if label["label"] != "irrelevant" else "NONE",
                }
                for label in row["labels"]
            ]
            augmented.append({
                **row, "index": f"{row['index']}-p{variant}",
                "question": pair["question"], "answer": pair["answer"],
                "answer_full": pair["answer"], "labels": labels,
                "gptoss_paraphrased": True,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in augmented) + "\n"
    )
    print(
        f"wrote {len(augmented)} paraphrases from {len(rows)} rows; "
        f"failures={failures}; generation_seconds={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
