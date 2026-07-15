#!/usr/bin/env python3
"""Apply label-blind materiality guards to an audited reader cache."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.core import (
    STOPWORDS,
    lexical_relevance,
    normalize_text,
    split_sentences,
)


CONTRADICTION = "DECISIVE_CONTRADICTION"
CARDINAL_WORDS = {
    word: str(number)
    for number, word in enumerate((
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    ))
}
ORDINAL_WORDS = {
    word: str(number)
    for number, word in enumerate((
        "zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
        "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
        "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
        "eighteenth", "nineteenth", "twentieth",
    ))
}
NUMBER = re.compile(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)*(?:st|nd|rd|th)?%?", re.I)
WORD = re.compile(r"[A-Za-z]+")
VALUE_TOKEN = re.compile(r"\d+(?:[.,]\d+)*(?:st|nd|rd|th)?%?|[A-Za-z]+", re.I)
SOURCE_MARKER = "\nSource sentence:"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def typed_salient_values(text: str) -> set[tuple[str, str]]:
    """Extract normalized values while preserving coarse semantic type."""
    values: set[tuple[str, str]] = set()
    for match in NUMBER.finditer(text):
        raw = match.group(0).casefold().replace(",", "")
        digits = re.match(r"\d+", raw)
        assert digits is not None
        number = digits.group(0)
        if raw.endswith("%"):
            kind = "percent"
        elif re.fullmatch(r"\d+(?:st|nd|rd|th)", raw):
            kind = "ordinal"
        elif len(number) == 4 and 1500 <= int(number) <= 2200:
            kind = "year"
        else:
            kind = "cardinal"
        values.add((kind, number if kind in {"ordinal", "cardinal"} else raw))
    for word in WORD.findall(text):
        normalized = word.casefold()
        if normalized in CARDINAL_WORDS:
            values.add(("cardinal", CARDINAL_WORDS[normalized]))
        elif normalized in ORDINAL_WORDS:
            values.add(("ordinal", ORDINAL_WORDS[normalized]))
    return values


def salient_values(text: str) -> set[str]:
    """Return a stable, human-readable projection used in audit reports/tests."""
    return {f"{kind}:{value}" for kind, value in typed_salient_values(text)}


def value_context(text: str, target: tuple[str, str]) -> set[str]:
    """Collect content tokens from sentences containing a normalized value."""
    context: set[str] = set()
    sentences = split_sentences(text, minimum_chars=1) or [normalize_text(text)]
    for sentence in sentences:
        if target not in typed_salient_values(sentence):
            continue
        for neighbor in VALUE_TOKEN.findall(sentence):
            normalized = neighbor.casefold()
            if normalized not in STOPWORDS and not typed_salient_values(neighbor):
                context.add(normalized)
    return context


def passage_parts(passage: dict[str, Any]) -> tuple[str, str]:
    text = str(passage.get("text") or "")
    claim_part, marker, source = text.partition(SOURCE_MARKER)
    if not marker:
        return "", normalize_text(text)
    claim = claim_part.removeprefix("Claim:").strip()
    return normalize_text(claim), normalize_text(source)


def repeats_disputed_question_value(
    question: str,
    proposition: str,
    source: str,
) -> bool:
    """Detect a contradiction caused by copying a disputed question value.

    This is intentionally narrow: the value must occur in both the question and
    assistant proposition, while being absent from the cited source. It does not
    use the deception label or attempt to infer whether the mistake was honest.
    """
    question_values = typed_salient_values(question)
    proposition_values = typed_salient_values(proposition)
    source_values = typed_salient_values(source)
    copied_missing = (question_values & proposition_values) - source_values
    for missing in copied_missing:
        kind, _ = missing
        proposition_of_kind = {value for value_kind, value in proposition_values if value_kind == kind}
        source_of_kind = {
            (value_kind, value)
            for value_kind, value in source_values
            if value_kind == kind
        }
        # A same-type value must actually be offered as a correction. If another
        # proposition value of that type is supported, the missing value is more
        # likely an anchor (for example "the next transit after 2012 is 2117").
        if not source_of_kind or proposition_of_kind & {value for _, value in source_of_kind}:
            continue
        claim_context = value_context(proposition, missing)
        for replacement in source_of_kind:
            if len(claim_context & value_context(source, replacement)) >= 2:
                return True
    return False


def focus_source(passage: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Trim an evidence window to the sentence most relevant to its claim."""
    proposition, source = passage_parts(passage)
    sentences = split_sentences(source, minimum_chars=1)
    if not proposition or len(sentences) <= 1:
        return passage, False
    _, best = max(
        enumerate(sentences),
        key=lambda item: (lexical_relevance(proposition, item[1]), -item[0]),
    )
    if normalize_text(best) == source:
        return passage, False
    updated = dict(passage)
    updated["text"] = f"Claim: {proposition}\nSource sentence: {normalize_text(best)}"
    return updated, True


def filter_passages(
    passages: list[dict[str, Any]],
    question: str,
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Filter one row without consulting labels or reader predictions."""
    stats: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for passage in passages:
        proposition, source = passage_parts(passage)
        if (
            passage.get("audit_relation") == CONTRADICTION
            and repeats_disputed_question_value(question, proposition, source)
        ):
            stats["dropped_question_value_contradiction"] += 1
            continue
        candidate = passage
        if mode in {"premise_focused", "premise_single"}:
            candidate, changed = focus_source(candidate)
            stats["focused_source_windows"] += int(changed)
        kept.append(candidate)

    if mode == "premise_single" and kept:
        first_claim = min(int(passage.get("claim_index", 0)) for passage in kept)
        selected = [
            passage
            for passage in kept
            if int(passage.get("claim_index", 0)) == first_claim
        ]
        stats["dropped_later_claims"] += len(kept) - len(selected)
        kept = selected
    return kept, stats


def question_map(rows: list[dict[str, Any]]) -> dict[tuple[str, Any], str]:
    output: dict[tuple[str, Any], str] = {}
    for row in rows:
        key = str(row["dataset"]), row["index"]
        question = normalize_text(str(row.get("question") or ""))
        if key in output and output[key] != question:
            raise ValueError(f"inconsistent questions for {key}")
        output[key] = question
    return output


def cross_dataset_donors(
    grouped: dict[tuple[str, Any], list[dict[str, Any]]],
) -> dict[tuple[str, Any], tuple[str, Any]]:
    """Match each active row to deterministic evidence from another dataset."""
    active = sorted(grouped)
    donors: dict[tuple[str, Any], tuple[str, Any]] = {}
    for position, key in enumerate(active):
        for offset in range(len(active)):
            candidate = active[(position + len(active) // 2 + offset) % len(active)]
            if candidate != key and candidate[0] != key[0]:
                donors[key] = candidate
                break
        if key not in donors:
            raise ValueError(f"no cross-dataset donor for {key}")
    return donors


def build_filtered_cache(
    reference_rows: list[dict[str, Any]],
    question_rows: list[dict[str, Any]],
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in {"premise_guard", "premise_focused", "premise_single"}:
        raise ValueError(f"unsupported mode: {mode}")
    questions = question_map(question_rows)
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    totals: Counter[str] = Counter()
    for row in reference_rows:
        key = str(row["dataset"]), row["index"]
        passages = list(row.get("real_passages") or row.get("passages") or [])
        if passages and key not in questions:
            raise ValueError(f"missing question for active row {key}")
        filtered, stats = filter_passages(passages, questions.get(key, ""), mode=mode)
        totals.update(stats)
        if filtered:
            grouped[key] = filtered
    donors = cross_dataset_donors(grouped)

    output = []
    for row in reference_rows:
        key = str(row["dataset"]), row["index"]
        real = grouped.get(key, [])
        donor = donors.get(key)
        output.append({
            "dataset": key[0],
            "index": key[1],
            "passages": real,
            "real_passages": real,
            "shuffled_passages": grouped[donor] if donor is not None else [],
            "shuffled_donor_dataset": donor[0] if donor is not None else None,
            "shuffled_donor_index": donor[1] if donor is not None else None,
        })
    summary = {
        "mode": mode,
        "rows": len(output),
        "active_rows": len(grouped),
        "passages": sum(len(passages) for passages in grouped.values()),
        "reference_characters": sum(
            len(str(passage.get("text") or ""))
            for passages in grouped.values()
            for passage in passages
        ),
        **dict(totals),
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--question-cache", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["premise_guard", "premise_focused", "premise_single"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    output, summary = build_filtered_cache(
        read_jsonl(args.reference_cache),
        read_jsonl(args.question_cache),
        mode=args.mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
