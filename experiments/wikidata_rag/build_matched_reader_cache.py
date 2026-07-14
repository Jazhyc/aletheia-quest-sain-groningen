#!/usr/bin/env python3
"""Build matched evidence-use SFT records from reviewed teacher caches."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import (
    append_reference_material,
)
from experiments.privileged_information_distillation.evaluate_student_sft import (
    load_retrieval_cache,
)


Key = tuple[str, Any]
REFERENCE_MARKER = "\n\n<reference_material>\n"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def usable_teacher_records(path: Path) -> dict[Key, dict[str, Any]]:
    records = {}
    for record in load_jsonl(path):
        if (
            record.get("parse_error")
            or record.get("label_match") is not True
            or not record.get("student_target")
        ):
            continue
        key = (str(record["dataset"]), record["index"])
        if key in records:
            raise ValueError(f"duplicate usable teacher key: {key}")
        records[key] = record
    return records


def prompt_without_reference(student_prompt: str) -> str:
    prefix, marker, _ = student_prompt.partition(REFERENCE_MARKER)
    if not marker:
        raise ValueError("evidence-aware student prompt lacks reference material")
    return prefix


def candidate_reference(candidate: dict[str, Any]) -> str:
    subject = str(candidate.get("subject", "Unknown subject"))
    qid = str(candidate.get("qid", "")).strip()
    title = f"{subject} ({qid})" if qid else subject
    fact = str(candidate.get("fact", "")).strip()
    return f"- {title}: {fact}"


def clone_variant(
    prompt_source: dict[str, Any],
    target_source: dict[str, Any],
    *,
    reference: str,
    variant: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_prompt = prompt_without_reference(str(prompt_source["student_prompt"]))
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
        "matched_cache_source": True,
    }


def donor_keys(keys: list[Key]) -> dict[Key, Key]:
    """Choose deterministic label-blind donors from a different dataset unit."""
    donors = {}
    for position, key in enumerate(keys):
        for offset in range(len(keys)):
            candidate = keys[(position + len(keys) // 2 + offset) % len(keys)]
            if candidate != key and candidate[0] != key[0]:
                donors[key] = candidate
                break
        if key not in donors:
            raise ValueError(f"no cross-dataset donor available for {key}")
    return donors


def build_variants(
    baseline: dict[Key, dict[str, Any]],
    evidence: dict[Key, dict[str, Any]],
    references: dict[Key, str],
    utility: dict[Key, dict[str, Any]],
    *,
    minimum_gain: float,
) -> list[dict[str, Any]]:
    keys = sorted(set(baseline) & set(evidence) & set(references) & set(utility))
    if not keys:
        raise ValueError("teacher/retrieval/utility caches have no common rows")
    donors = donor_keys(keys)
    variants = []
    for key in keys:
        evidence_record = evidence[key]
        baseline_record = baseline[key]
        utility_record = utility[key]
        labels = {
            int(evidence_record["label"]),
            int(baseline_record["label"]),
            int(utility_record["label"]),
        }
        if len(labels) != 1:
            raise ValueError(f"label mismatch across matched sources for {key}: {labels}")
        variants.append({
            **evidence_record,
            "evidence_variant": "retrieved_real",
            "evidence_metadata": {"reference_source": "frozen_broad_index"},
            "matched_cache_source": True,
        })
        variants.append(clone_variant(
            evidence_record,
            baseline_record,
            reference="",
            variant="empty",
        ))
        donor = donors[key]
        variants.append(clone_variant(
            evidence_record,
            baseline_record,
            reference=references[donor],
            variant="shuffled",
            metadata={"donor_dataset": donor[0], "donor_index": donor[1]},
        ))

        candidates = list(utility_record.get("candidates") or [])
        harmful = [
            candidate for candidate in candidates
            if float(candidate.get("controlled_utility", 0.0)) < -minimum_gain
            and candidate.get("semantic_label") != "decisive"
        ]
        if harmful:
            selected = min(harmful, key=lambda item: float(item["controlled_utility"]))
            variants.append(clone_variant(
                evidence_record,
                baseline_record,
                reference=candidate_reference(selected),
                variant="reader_harmful",
                metadata={
                    "candidate_id": selected.get("id"),
                    "controlled_utility": selected["controlled_utility"],
                    "semantic_label": selected.get("semantic_label"),
                },
            ))

        helpful = [
            candidate for candidate in candidates
            if float(candidate.get("controlled_utility", 0.0)) > minimum_gain
        ]
        if helpful:
            selected = max(helpful, key=lambda item: float(item["controlled_utility"]))
            variants.append(clone_variant(
                evidence_record,
                baseline_record,
                reference=candidate_reference(selected),
                variant="reader_helpful",
                metadata={
                    "candidate_id": selected.get("id"),
                    "controlled_utility": selected["controlled_utility"],
                    "semantic_label": selected.get("semantic_label"),
                },
            ))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-teacher", type=Path, required=True)
    parser.add_argument("--evidence-teacher", type=Path, required=True)
    parser.add_argument("--retrieval-cache", type=Path, required=True)
    parser.add_argument("--utility-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.01)
    args = parser.parse_args()

    baseline = usable_teacher_records(args.baseline_teacher)
    evidence = usable_teacher_records(args.evidence_teacher)
    references = load_retrieval_cache(args.retrieval_cache)
    utility_rows = load_jsonl(args.utility_cache)
    utility = {
        (str(record["dataset"]), record["index"]): record
        for record in utility_rows
    }
    if len(utility) != len(utility_rows):
        raise ValueError("utility cache has duplicate dataset/index keys")
    variants = build_variants(
        baseline, evidence, references, utility,
        minimum_gain=args.minimum_gain,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in variants) + "\n"
    )
    counts = Counter(str(record["evidence_variant"]) for record in variants)
    labels = Counter(int(record["label"]) for record in variants)
    print(json.dumps({
        "rows": len(variants),
        "source_rows": len({(row["dataset"], row["index"]) for row in variants}),
        "variants": counts,
        "labels": labels,
        "output": args.output.as_posix(),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
