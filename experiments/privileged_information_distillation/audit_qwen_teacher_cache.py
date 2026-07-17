#!/usr/bin/env python3
"""Strictly audit, preserve, and merge sharded Qwen teacher caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import (
    format_student_target,
    parse_teacher_target,
    split_qwen_think_completion,
)


LEAK_PATTERNS = {
    "private_marker": re.compile(r"</?think>", re.IGNORECASE),
    "ground_truth": re.compile(r"\bground[- ]truth\b", re.IGNORECASE),
    "privileged_information": re.compile(
        r"\bprivileged (?:information|label)\b", re.IGNORECASE
    ),
    "teacher_meta": re.compile(
        r"\b(?:as (?:a|the) teacher|being a teacher)\b", re.IGNORECASE
    ),
    "instruction_meta": re.compile(r"\bthese instructions\b", re.IGNORECASE),
}


def record_key(record: dict[str, Any]) -> str:
    return json.dumps(
        [record.get("dataset"), record.get("index")],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def record_hash(record: dict[str, Any]) -> str:
    encoded = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"teacher shard is missing: {path}")
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise RuntimeError(f"non-object record at {path}:{line_number}")
            records.append(record)
    return records


def audit_records(
    records: list[dict[str, Any]],
    *,
    expected_total: int,
    minimum_usable: int,
    allow_unclosed: bool,
    allow_truncated_targets: bool,
    maximum_label_imbalance: int | None,
    expected_model: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    manifest: dict[str, str] = {}
    label_counts: Counter[int] = Counter()

    if len(records) != expected_total:
        errors.append(f"expected {expected_total} records, found {len(records)}")

    for record in records:
        key = record_key(record)
        if key in seen:
            errors.append(f"duplicate record {key}")
            continue
        seen.add(key)
        label = record.get("label")
        if label not in (0, 1):
            errors.append(f"{key}: invalid label {label!r}")
            continue
        label = int(label)
        if record.get("teacher_model") != expected_model:
            errors.append(
                f"{key}: unexpected teacher_model={record.get('teacher_model')!r}"
            )
        if record.get("teacher_output_format") != "qwen_think":
            errors.append(
                f"{key}: unexpected teacher_output_format="
                f"{record.get('teacher_output_format')!r}"
            )

        raw = str(record.get("raw_completion") or "")
        split = split_qwen_think_completion(raw)
        parse_error = bool(record.get("parse_error"))
        if split is None:
            counts["unclosed"] += 1
            if not allow_unclosed:
                errors.append(f"{key}: missing </think> closure")
            if not parse_error:
                errors.append(f"{key}: unclosed completion is marked usable")
            if record.get("student_target") or record.get("teacher_final"):
                errors.append(f"{key}: unclosed completion exposes a visible target")
            continue

        if parse_error:
            visible_final = split[1]
            is_truncated_target = (
                visible_final.lstrip().startswith("<reasoning_summary>")
                and "</reasoning_summary>" not in visible_final
            )
            if is_truncated_target:
                counts["truncated_target"] += 1
                if not allow_truncated_targets:
                    errors.append(f"{key}: truncated visible target is not allowed")
                if record.get("teacher_final") != visible_final:
                    errors.append(f"{key}: inconsistent truncated teacher_final")
                if any(
                    record.get(field) is not None
                    for field in ("reasoning_summary", "prediction", "student_target")
                ):
                    errors.append(f"{key}: truncated target exposes parsed fields")
                if record.get("label_match") is not False:
                    errors.append(f"{key}: truncated target has invalid label_match")
            else:
                errors.append(f"{key}: closed completion failed target parsing")
            continue
        parsed = parse_teacher_target(
            raw,
            expected_prediction=label,
            output_format="qwen_think",
        )
        if parsed is None:
            errors.append(f"{key}: cache is usable but strict reparse failed")
            continue
        summary, prediction = parsed
        expected_target = format_student_target(summary, prediction)
        visible_final = split[1]
        checks = {
            "prediction": record.get("prediction") == prediction == label,
            "label_match": record.get("label_match") is True,
            "reasoning_summary": record.get("reasoning_summary") == summary,
            "student_target": record.get("student_target") == expected_target,
            "teacher_final": record.get("teacher_final") == visible_final,
        }
        for name, passed in checks.items():
            if not passed:
                errors.append(f"{key}: inconsistent {name}")
        visible = "\n".join(
            str(record.get(field) or "")
            for field in ("teacher_final", "student_target")
        )
        for name, pattern in LEAK_PATTERNS.items():
            if pattern.search(visible):
                counts[f"leak_{name}"] += 1
                errors.append(f"{key}: visible target matches leakage rule {name}")
        counts["usable"] += 1
        label_counts[label] += 1
        manifest[key] = record_hash(record)

    usable = counts["usable"]
    if usable < minimum_usable:
        errors.append(f"usable target gate failed: {usable} < {minimum_usable}")
    imbalance = abs(label_counts[0] - label_counts[1])
    if maximum_label_imbalance is not None and imbalance > maximum_label_imbalance:
        errors.append(
            f"usable label imbalance gate failed: {imbalance} > {maximum_label_imbalance}"
        )

    report = {
        "expected_total": expected_total,
        "records": len(records),
        "unique_records": len(seen),
        "usable": usable,
        "unclosed": counts["unclosed"],
        "truncated_targets": counts["truncated_target"],
        "usable_by_label": {"0": label_counts[0], "1": label_counts[1]},
        "usable_label_imbalance": imbalance,
        "leak_counts": {
            name.removeprefix("leak_"): count
            for name, count in sorted(counts.items())
            if name.startswith("leak_")
        },
        "errors": errors,
        "passed": not errors,
    }
    return report, manifest


def verify_manifest(manifest: dict[str, str], path: Path) -> None:
    expected = json.loads(path.read_text())
    changed = [key for key, digest in expected.items() if manifest.get(key) != digest]
    if changed:
        preview = ", ".join(changed[:5])
        raise RuntimeError(
            f"{len(changed)} previously usable records changed or disappeared: {preview}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--expected-total", required=True, type=int)
    parser.add_argument("--minimum-usable", required=True, type=int)
    parser.add_argument("--allow-unclosed", action="store_true")
    parser.add_argument("--allow-truncated-targets", action="store_true")
    parser.add_argument("--maximum-label-imbalance", type=int)
    parser.add_argument("--expected-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.shard)
    report, manifest = audit_records(
        records,
        expected_total=args.expected_total,
        minimum_usable=args.minimum_usable,
        allow_unclosed=args.allow_unclosed,
        allow_truncated_targets=args.allow_truncated_targets,
        maximum_label_imbalance=args.maximum_label_imbalance,
        expected_model=args.expected_model,
    )
    if args.verify_manifest:
        try:
            verify_manifest(manifest, args.verify_manifest)
        except RuntimeError as error:
            report["errors"].append(str(error))
            report["passed"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError(f"teacher cache audit failed; see {args.output}")
    if args.verify_manifest:
        print(f"verified immutable records from {args.verify_manifest}")
    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    if args.merged_output:
        args.merged_output.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(records, key=record_key)
        with args.merged_output.open("w") as handle:
            for record in ordered:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {args.merged_output}")


if __name__ == "__main__":
    main()
