#!/usr/bin/env python3
"""Recover high-precision polarity from frozen decisive retrieval candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time
from typing import Any

from experiments.privileged_information_distillation.core import extract_harmony_final


COMPARISONS = {
    "same_value",
    "alias_or_equivalent",
    "mutually_exclusive",
    "compatible_additional",
    "unknown",
}
POLAR_COMPARISONS = {
    "same_value": "supports",
    "alias_or_equivalent": "supports",
    "mutually_exclusive": "contradicts",
}
# A different value for these predicates is not normally a contradiction because
# Wikidata can legitimately store several values. Keep support twins, but veto
# contradiction targets unless a later relation-specific audit replaces this list.
NON_EXCLUSIVE_PREDICATES = {
    "award received",
    "cast member",
    "characters",
    "child",
    "has part",
    "member of",
    "member of sports team",
    "notable work",
    "occupation",
    "participant",
    "position held",
    "spouse",
    "subsidiary",
}


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decisive_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten only candidates that passed the frozen polarity-blind teacher."""
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        labels = {item["id"]: item["label"] for item in row.get("labels", [])}
        for candidate in row.get("candidates", []):
            if labels.get(candidate["id"]) != "decisive":
                continue
            records.append(
                {
                    "source_row_index": row_index,
                    "question_group": row["question_group"],
                    "question": row["question"],
                    "answer_full": row["answer_full"],
                    "candidate": candidate,
                    # Retained for offline analysis, never rendered to the teacher.
                    "deceptive": row.get("deceptive"),
                    "dataset_name": row.get("dataset_name"),
                }
            )
    return records


def prompt_for(record: dict[str, Any]) -> str:
    candidate = record["candidate"]
    return f"""You are the second stage of a factual-evidence annotation pipeline.
An earlier, polarity-blind audit said the database candidate is sufficient to
settle the question's direct answer. Recheck that assumption and compare values.
Never infer deception, intent, or the hidden class label.

The RESPONSE and DATABASE EVIDENCE are separate sources. claim_quote and
claimed_value must be copied from RESPONSE, never from DATABASE EVIDENCE.
database_value must be copied from either the database subject or database fact.

Return exactly one JSON object with these fields:
- id: copy the candidate id
- claim_quote: exact contiguous response substring containing the direct answer
- claimed_value: exact non-empty response substring naming its answer value
- database_value: exact non-empty database substring naming the corresponding value
- entity_relation_match: true only when the evidence concerns the same entity/sense
  and the exact relation asked by the question
- comparison: exactly one of same_value, alias_or_equivalent, mutually_exclusive,
  compatible_additional, unknown
- reason: one concise sentence

Use same_value for textually identical values and alias_or_equivalent only for
genuine aliases (for example Hannover/Hanover), not merely related concepts.
Use mutually_exclusive only when both values answer the same requested slot and
cannot both be correct in context. Multi-valued relations, broader/narrower
classes, partial dates, hierarchy, and facts needing outside knowledge are
compatible_additional or unknown. If the earlier audit was wrong, set
entity_relation_match false or comparison unknown. Do not output a polarity label.

QUESTION:
{record["question"]}

RESPONSE:
{record["answer_full"][:1800]}

DATABASE EVIDENCE:
id: {candidate["id"]}
subject: {candidate["subject"]}
predicate: {candidate["predicate"]}
fact: {candidate["fact"]}
"""


def parse_json_object(completion: str) -> dict[str, Any] | None:
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
            if isinstance(value, dict):
                return value
    return None


def exact_span(span: str, source: str) -> bool:
    return bool(span.strip()) and span in source


def validate_annotation(
    parsed: dict[str, Any] | None,
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate source provenance and derive polarity without trusting a label."""
    if parsed is None:
        return None, ["json_parse"]
    candidate = record["candidate"]
    errors: list[str] = []
    candidate_id = str(parsed.get("id", ""))
    claim_quote = str(parsed.get("claim_quote", ""))
    claimed_value = str(parsed.get("claimed_value", ""))
    database_value = str(parsed.get("database_value", ""))
    comparison = str(parsed.get("comparison", "")).lower()
    entity_relation_match = parsed.get("entity_relation_match")
    database_source = f'{candidate["subject"]}\n{candidate["fact"]}'

    if candidate_id != candidate["id"]:
        errors.append("id")
    if not exact_span(claim_quote, record["answer_full"]):
        errors.append("claim_quote")
    if not exact_span(claimed_value, claim_quote):
        errors.append("claimed_value")
    if not exact_span(database_value, database_source):
        errors.append("database_value")
    if entity_relation_match not in {True, False}:
        errors.append("entity_relation_match")
    if comparison not in COMPARISONS:
        errors.append("comparison")

    polarity = (
        POLAR_COMPARISONS.get(comparison)
        if entity_relation_match is True and not errors
        else None
    )
    polarity_veto = None
    if polarity == "contradicts" and candidate["predicate"] in NON_EXCLUSIVE_PREDICATES:
        polarity = None
        polarity_veto = "nonexclusive_predicate"
    annotation = {
        "id": candidate_id,
        "claim_quote": claim_quote,
        "claimed_value": claimed_value,
        "database_value": database_value,
        "entity_relation_match": entity_relation_match,
        "comparison": comparison,
        "polarity": polarity,
        "polarity_veto": polarity_veto,
        "reason": str(parsed.get("reason", "")).strip(),
    }
    return annotation, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-input", type=Path)
    parser.add_argument("--extra-output", type=Path)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if bool(args.extra_input) != bool(args.extra_output):
        parser.error("--extra-input and --extra-output must be provided together")
    primary = decisive_candidates(load(args.input))
    extra = decisive_candidates(load(args.extra_input)) if args.extra_input else []
    if args.limit is not None:
        primary = primary[: args.limit]
        extra = extra[: args.limit]
    primary_count = len(primary)
    records = primary + extra

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_for(record)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for record in records
    ]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
    )
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    started = time.time()
    outputs = []
    for start in range(0, len(prompts), args.batch_size):
        outputs.extend(llm.generate(prompts[start : start + args.batch_size], sampling))
    elapsed = time.time() - started

    annotated = []
    for record, prompt, output in zip(records, prompts, outputs, strict=True):
        completion = output.outputs[0].text if output.outputs else ""
        annotation, errors = validate_annotation(parse_json_object(completion), record)
        annotated.append(
            {
                **record,
                "teacher_model": args.model,
                "rendered_prompt": prompt,
                "raw_completion": completion,
                "annotation": annotation,
                "validation_errors": errors,
            }
        )

    output_groups = [(args.output, annotated[:primary_count])]
    if args.extra_output:
        output_groups.append((args.extra_output, annotated[primary_count:]))
    for output_path, output_records in output_groups:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in output_records) + "\n"
        )
    counts = Counter(
        (row["annotation"] or {}).get("polarity") or "abstain" for row in annotated
    )
    print(
        f"wrote {len(annotated)} candidates in {elapsed:.1f}s; "
        f"polarity={dict(counts)}"
    )


if __name__ == "__main__":
    main()
