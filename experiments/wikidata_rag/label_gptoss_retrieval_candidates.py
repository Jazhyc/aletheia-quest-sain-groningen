#!/usr/bin/env python3
"""Use local GPT-OSS to supervise claim-to-Wikidata relevance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any

from experiments.privileged_information_distillation.core import extract_harmony_final
from experiments.wikidata_rag.claim_retrieval import normalize


LEGACY_LABELS = {"supports", "contradicts", "relevant_insufficient", "irrelevant"}
DECISIVENESS_LABELS = {"decisive", "relevant_insufficient", "irrelevant"}
STRUCTURAL_ERROR_PREFIXES = {"json_parse", "duplicate", "unknown", "label", "missing"}


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompt_for(row: dict[str, Any], label_mode: str = "polarity") -> str:
    facts = "\n".join(
        f"{item['id']} | {item['subject']} | {item['fact']}"
        for item in row["candidates"]
    )
    if label_mode == "decisive":
        return f"""You are labeling retrieval evidence for a factual question-answer checker.

First identify the response's direct answer to the question. For every candidate,
ask whether that fact BY ITSELF settles whether the direct answer is correct. Ignore
whether the fact agrees or disagrees with the response: both are decisive. Do not
use deception labels or infer intent.

- decisive: the fact directly supplies the requested subject, relation, and value,
  so it settles the direct answer without outside knowledge or an unstated bridge.
- relevant_insufficient: it concerns the correct entity/topic but has the wrong
  relation or granularity, gives only partial evidence, or needs an outside inference.
- irrelevant: it concerns the wrong entity/topic or an unrelated answer slot.

Important examples:
- If the answer names Arthur but the fact directly says the author is Wilkie, the
  fact is decisive. Agreement versus contradiction is deliberately not labeled.
- A cast-member fact does not settle who sang a song: relevant_insufficient.
- A country's name does not settle its vehicle registration code: relevant_insufficient.
- A date or occupation about the right entity does not settle its name or creator:
  relevant_insufficient.
- A direct "instance of nuclear plant" fact is decisive for nuclear versus coal.

Return one JSON object per candidate, in the same order, inside a JSON array. Each
object must contain only id, label, and a concise reason. Use only the three exact
labels above.

Question:
{row['question']}

Response:
{row['answer_full'][:1800]}

Candidate facts:
{facts}
"""
    return f"""You are labeling retrieval evidence for a factual question-answer checker.

First identify the response's direct answer to the question. Judge every candidate
only against that requested answer slot, not against incidental true details in the
response. Do not use deception labels or infer intent.

Apply this strict logical test: the candidate fact by itself, without outside
knowledge or an unstated bridge, must contain the subject, relation, and value needed
to establish or refute the direct answer.

- supports: the fact alone entails the direct answer.
- contradicts: the fact alone entails that the direct answer is false.
- relevant_insufficient: correct topic/entity but wrong relation, wrong granularity,
  partial evidence, or an outside inference is still needed.
- irrelevant: wrong entity/topic or unrelated to the requested answer.

Examples of relevant_insufficient rather than decisive evidence: cast membership
does not prove who sang a song; a city does not directly answer a state question;
instance-of a broad class does not prove a specific identity; `different from X`
does not mean X cannot be a member/value; a supporting side detail does not settle
the answer requested by the question.

For supports, contradicts, or relevant_insufficient, copy an exact non-empty quote
from the response into claim_quote. For irrelevant use claim_quote "NONE". Do not
invent or correct Wikidata values. Return one JSON object per candidate, in the
same order, inside a JSON array. Each object must contain only id, label,
claim_quote, and a concise reason.

Question:
{row['question']}

Response:
{row['answer_full'][:1800]}

Candidate facts:
{facts}
"""


def parse_json_array(completion: str) -> list[dict[str, Any]] | None:
    final = extract_harmony_final(completion)
    starts = [match.start() for match in re.finditer(r"\[", final)]
    for start in starts:
        for end in range(len(final), start, -1):
            if final[end - 1] != "]":
                continue
            try:
                value = json.loads(final[start:end])
            except json.JSONDecodeError:
                continue
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
    return None


def grounded_quote(quote: str, answer: str) -> bool:
    quote_norm = normalize(quote)
    return bool(quote_norm) and quote_norm in normalize(answer)


def validate_labels(
    parsed: list[dict[str, Any]] | None,
    row: dict[str, Any],
    label_mode: str = "polarity",
) -> tuple[list[dict[str, Any]], list[str]]:
    if parsed is None:
        return [], ["json_parse"]
    expected = [item["id"] for item in row["candidates"]]
    errors: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    allowed_labels = DECISIVENESS_LABELS if label_mode == "decisive" else LEGACY_LABELS
    for item in parsed:
        candidate_id = str(item.get("id", ""))
        label = str(item.get("label", "")).lower()
        quote = str(item.get("claim_quote", ""))
        if candidate_id in by_id:
            errors.append(f"duplicate:{candidate_id}")
            continue
        if candidate_id not in expected:
            errors.append(f"unknown:{candidate_id}")
            continue
        if label not in allowed_labels:
            errors.append(f"label:{candidate_id}")
            continue
        record = {
            "id": candidate_id, "label": label,
            "reason": str(item.get("reason", "")).strip(),
        }
        if label_mode != "decisive":
            if label == "irrelevant":
                if quote.upper() != "NONE":
                    errors.append(f"irrelevant_quote:{candidate_id}")
            elif not grounded_quote(quote, row["answer_full"]):
                errors.append(f"ungrounded_quote:{candidate_id}")
            record["claim_quote"] = quote
        by_id[candidate_id] = record
    missing = [candidate_id for candidate_id in expected if candidate_id not in by_id]
    errors.extend(f"missing:{candidate_id}" for candidate_id in missing)
    return [by_id[candidate_id] for candidate_id in expected if candidate_id in by_id], errors


def diverse_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    rows = [row for row in rows if row.get("candidates")]
    rows.sort(key=lambda row: (bool(row.get("currently_covered")), row["question_group"]))
    if limit is None:
        return rows
    selected = []
    seen_groups = set()
    for row in rows:
        group = row["question_group"]
        if group in seen_groups:
            continue
        seen_groups.add(group)
        selected.append(row)
        if len(selected) == limit:
            break
    return selected


def structural_errors(errors: list[str]) -> list[str]:
    return [
        error for error in errors
        if error.partition(":")[0] in STRUCTURAL_ERROR_PREFIXES
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-input", type=Path)
    parser.add_argument("--extra-output", type=Path)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidates-per-row", type=int, default=12)
    parser.add_argument("--label-mode", choices=("polarity", "decisive"), default="polarity")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if bool(args.extra_input) != bool(args.extra_output):
        parser.error("--extra-input and --extra-output must be provided together")
    primary_rows = diverse_rows(load(args.input), args.limit)
    extra_rows = diverse_rows(load(args.extra_input), args.limit) if args.extra_input else []
    primary_count = len(primary_rows)
    rows = primary_rows + extra_rows
    for row in rows:
        row["candidates"] = row["candidates"][:args.candidates_per_row]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_for(row, args.label_mode)}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    ) for row in rows]
    llm = LLM(
        model=args.model, dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=0.9, max_model_len=8192,
    )
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    started = time.time()
    outputs = []
    for start in range(0, len(prompts), args.batch_size):
        outputs.extend(llm.generate(prompts[start:start + args.batch_size], sampling))
    elapsed = time.time() - started

    records = []
    for row, prompt, output in zip(rows, prompts, outputs, strict=True):
        completion = output.outputs[0].text if output.outputs else ""
        labels, errors = validate_labels(parse_json_array(completion), row, args.label_mode)
        records.append({
            **row, "teacher_model": args.model, "rendered_prompt": prompt,
            "raw_completion": completion, "labels": labels, "label_mode": args.label_mode,
            "parse_error": bool(structural_errors(errors)),
            "grounding_error": bool(errors) and not bool(structural_errors(errors)),
            "validation_errors": errors,
        })
    output_groups = [(args.output, records[:primary_count])]
    if args.extra_output:
        output_groups.append((args.extra_output, records[primary_count:]))
    for output_path, output_records in output_groups:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in output_records) + "\n"
        )
    parsed = sum(not row["parse_error"] for row in records)
    destinations = ", ".join(str(path) for path, _ in output_groups)
    print(f"wrote {len(records)} rows ({parsed} valid) in {elapsed:.1f}s to {destinations}")


if __name__ == "__main__":
    main()
