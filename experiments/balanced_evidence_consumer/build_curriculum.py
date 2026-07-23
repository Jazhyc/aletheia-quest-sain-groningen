#!/usr/bin/env python3
"""Compose balanced evidence-use SFT records from reviewed teacher artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import (
    REFERENCE_PREAMBLE,
    append_reference_material,
)


Key = tuple[str, Any]
REFERENCE_START = "\n\n<reference_material>\n"
REFERENCE_END = "\n</reference_material>"
DECISIVE_CONTRADICTION = "DECISIVE_CONTRADICTION"
NON_DECISIVE = {"TOPICAL", "INSUFFICIENT"}
PREDICTION_RE = re.compile(r"Prediction:\s*([01])\s*$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def usable_records(path: Path) -> dict[Key, dict[str, Any]]:
    records: dict[Key, dict[str, Any]] = {}
    for record in read_jsonl(path):
        if (
            record.get("parse_error")
            or record.get("label_match") is not True
            or not record.get("student_target")
            or "varied-deception" not in str(record.get("dataset") or "")
        ):
            continue
        key = (str(record["dataset"]), record["index"])
        if key in records:
            raise ValueError(f"duplicate usable teacher key: {key}")
        records[key] = record
    return records


def split_reference(student_prompt: str) -> tuple[str, str]:
    prefix, marker, tail = str(student_prompt).partition(REFERENCE_START)
    if not marker:
        raise ValueError("evidence-aware prompt lacks reference material")
    reference, end, remainder = tail.partition(REFERENCE_END)
    if not end or remainder.strip():
        raise ValueError("malformed reference material block")
    reference = reference.strip()
    if reference.startswith(REFERENCE_PREAMBLE):
        reference = reference[len(REFERENCE_PREAMBLE):].strip()
    return prefix, reference


def clone_variant(
    prompt_source: dict[str, Any],
    target_source: dict[str, Any],
    *,
    reference: str,
    variant: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_prompt, _ = split_reference(str(prompt_source["student_prompt"]))
    return {
        **target_source,
        "dataset": prompt_source["dataset"],
        "index": prompt_source["index"],
        "label": prompt_source["label"],
        "student_prompt": append_reference_material(base_prompt, reference),
        "evidence_variant": variant,
        "evidence_metadata": metadata or {},
        "teacher_prompt": None,
        "raw_completion": None,
        "harmony_final": None,
        "balanced_evidence_curriculum": True,
    }


def with_evidence_assessment(
    record: dict[str, Any],
    assessment: str,
) -> dict[str, Any]:
    """Add the evidence-aware schema to a reviewed ordinary teacher target."""
    target = str(record["student_target"])
    prefix = "<reasoning_summary>\n"
    suffix = "\n</reasoning_summary>"
    if not target.startswith(prefix) or suffix not in target:
        raise ValueError("ordinary target does not follow reasoning-summary schema")
    summary, remainder = target[len(prefix):].split(suffix, 1)
    evidence_summary = (
        f"Evidence assessment: {assessment.strip()} "
        f"Decision: {summary.strip()}"
    )
    return {
        **record,
        "reasoning_summary": evidence_summary,
        "student_target": f"{prefix}{evidence_summary}{suffix}{remainder}",
    }


def insufficient_references(path: Path) -> dict[Key, str]:
    """Select one auditor-rejected candidate per row without using class labels."""
    by_key: dict[Key, list[tuple[int, str]]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("parse_valid"):
                continue
            candidates = list(row.get("candidates") or [])
            for assessment in row.get("candidate_assessments") or []:
                relation = str(assessment.get("relation") or "")
                if relation not in NON_DECISIVE:
                    continue
                candidate_id = int(assessment.get("candidate_id") or 0)
                if not 1 <= candidate_id <= len(candidates):
                    continue
                candidate = candidates[candidate_id - 1]
                proposition = str(row.get("proposition") or "").strip()
                reference = (
                    f"- {candidate.get('title', 'Unknown source')}: "
                    f"Claim: {proposition}\n"
                    f"Source sentence: {str(candidate.get('text') or '').strip()}"
                )
                priority = 0 if relation == "TOPICAL" else 1
                key = (str(row["dataset"]), row["index"])
                by_key[key].append((priority, reference))
    return {
        key: min(candidates, key=lambda item: (item[0], len(item[1])))[1]
        for key, candidates in by_key.items()
    }


def contradiction_keys(path: Path) -> dict[int, list[Key]]:
    """Collect rows with auditor-selected contradictory evidence by class label."""
    keys: dict[int, set[Key]] = defaultdict(set)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("parse_valid")
                and row.get("decision") == DECISIVE_CONTRADICTION
                and row.get("selected_candidate")
            ):
                keys[int(row["label"])].add((str(row["dataset"]), row["index"]))
    return {label: sorted(values) for label, values in keys.items()}


def audit_variants(variants: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify variant identity, prompt isolation, and target consistency."""
    if not variants:
        raise ValueError("balanced evidence curriculum is empty")
    seen: set[tuple[str, Any, str]] = set()
    source_labels: dict[Key, int] = {}
    variant_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    max_prompt_chars = 0
    max_target_chars = 0
    for row in variants:
        key = (str(row["dataset"]), row["index"])
        label = int(row["label"])
        previous_label = source_labels.setdefault(key, label)
        if previous_label != label:
            raise ValueError(f"source label changes across variants for {key}")
        variant = str(row.get("evidence_variant") or "").strip()
        identity = (*key, variant)
        if not variant or identity in seen:
            raise ValueError(f"missing or duplicate evidence identity: {identity}")
        seen.add(identity)

        prompt = str(row.get("student_prompt") or "")
        target = str(row.get("student_target") or "")
        if prompt.count("<reference_material>") != 1:
            raise ValueError(f"expected one reference block for {identity}")
        if prompt.count("</reference_material>") != 1:
            raise ValueError(f"expected one closed reference block for {identity}")
        if "<ground_truth>" in prompt:
            raise ValueError(f"ground-truth tag leaked into student prompt: {identity}")
        if target.count("Evidence assessment:") != 1:
            raise ValueError(f"missing evidence assessment in target: {identity}")
        prediction = PREDICTION_RE.search(target)
        if prediction is None or int(prediction.group(1)) != label:
            raise ValueError(f"target prediction does not match label: {identity}")
        variant_counts[variant] += 1
        label_counts[label] += 1
        max_prompt_chars = max(max_prompt_chars, len(prompt))
        max_target_chars = max(max_target_chars, len(target))

    if abs(label_counts[0] - label_counts[1]) > 1:
        raise ValueError(f"curriculum labels are not balanced: {label_counts}")
    honest_errors = variant_counts["honest_ordinary_error_control"]
    deceptive_conflicts = variant_counts["deceptive_contradiction_control"]
    if honest_errors != deceptive_conflicts:
        raise ValueError(
            "ordinary-error and deceptive-contradiction controls are unbalanced: "
            f"{honest_errors} != {deceptive_conflicts}"
        )
    return {
        "rows": len(variants),
        "source_rows": len(source_labels),
        "variants": dict(sorted(variant_counts.items())),
        "labels": {str(label): count for label, count in sorted(label_counts.items())},
        "max_prompt_chars": max_prompt_chars,
        "max_target_chars": max_target_chars,
    }


def build_variants(
    baseline: dict[Key, dict[str, Any]],
    real: dict[Key, dict[str, Any]],
    shuffled: dict[Key, dict[str, Any]],
    insufficient: dict[Key, str],
    contradictions: dict[int, list[Key]],
) -> list[dict[str, Any]]:
    keys = sorted(set(baseline) & set(real) & set(shuffled))
    if not keys:
        raise ValueError("teacher artifacts have no common varied rows")
    variants = []
    for key in keys:
        labels = {int(source[key]["label"]) for source in (baseline, real, shuffled)}
        if len(labels) != 1:
            raise ValueError(f"label mismatch across teacher artifacts for {key}")
        real_base, real_reference = split_reference(str(real[key]["student_prompt"]))
        shuffled_base, shuffled_reference = split_reference(
            str(shuffled[key]["student_prompt"])
        )
        if real_base != shuffled_base:
            raise ValueError(f"real/shuffled prompt base mismatch for {key}")

        variants.append({
            **real[key],
            "evidence_variant": "real_decisive",
            "evidence_metadata": {},
            "balanced_evidence_curriculum": True,
        })
        variants.append(clone_variant(
            real[key],
            with_evidence_assessment(
                baseline[key],
                "No retrieved source is available, so the decision must rely on "
                "the conversation and final output.",
            ),
            reference="",
            variant="explicit_empty",
        ))
        variants.append({
            **shuffled[key],
            "evidence_variant": "shuffled_irrelevant",
            "evidence_metadata": {},
            "balanced_evidence_curriculum": True,
        })
        variants.append(clone_variant(
            real[key],
            real[key],
            reference=f"{real_reference}\n{shuffled_reference}",
            variant="mixed_relevant_irrelevant",
        ))
        if key in insufficient:
            variants.append(clone_variant(
                real[key],
                with_evidence_assessment(
                    baseline[key],
                    "The retrieved source is merely topical or insufficient to "
                    "settle the output claim and should be ignored.",
                ),
                reference=insufficient[key],
                variant="topical_or_insufficient",
            ))

    label0 = [key for key in contradictions.get(0, []) if key in real]
    label1 = [key for key in contradictions.get(1, []) if key in real]
    paired = min(len(label0), len(label1))
    for label, selected in ((0, label0[:paired]), (1, label1[:paired])):
        variant = (
            "honest_ordinary_error_control"
            if label == 0
            else "deceptive_contradiction_control"
        )
        for key in selected:
            variants.append({
                **real[key],
                "evidence_variant": variant,
                "evidence_metadata": {"balanced_contradiction_stratum": True},
                "balanced_evidence_curriculum": True,
            })
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-teacher", type=Path, required=True)
    parser.add_argument("--real-teacher", type=Path, required=True)
    parser.add_argument("--shuffled-teacher", type=Path, required=True)
    parser.add_argument("--audits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    baseline = usable_records(args.baseline_teacher)
    real = usable_records(args.real_teacher)
    shuffled = usable_records(args.shuffled_teacher)
    insufficient = insufficient_references(args.audits)
    contradictions = contradiction_keys(args.audits)
    variants = build_variants(
        baseline,
        real,
        shuffled,
        insufficient,
        contradictions,
    )
    report = audit_variants(variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in variants) + "\n"
    )
    report.update({
        "insufficient_source_rows": len(insufficient),
        "contradiction_rows": {
            str(label): len(keys) for label, keys in contradictions.items()
        },
        "output": args.output.as_posix(),
    })
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
