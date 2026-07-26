#!/usr/bin/env python3
"""Build full-trace, all-member Qwen-27B D/K/S distillation targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


MEMBER_MAP = {
    "details4096": "details3072_logit",
    "known4096": "known3072_logit",
    "scrutiny4096": "scrutiny3072_logit",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def keyed(
    records: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = tuple(record[field] for field in fields)
        if key in result:
            raise ValueError(f"duplicate record key: {key}")
        result[key] = record
    return result


def context_suffix(student_prompt: str) -> str:
    """Recover the exact rendered context/output block from a cached prompt."""
    marker = "\n\n<context>"
    _, separator, suffix = student_prompt.partition(marker)
    if not separator:
        raise ValueError("cached student prompt has no <context> block")
    return marker + suffix


def stable_train_keys(
    rows: list[dict[str, Any]],
    fraction: float,
    seed: int,
) -> set[tuple[str, Any]]:
    """Select whole rows within each dataset/label stratum."""
    strata: dict[tuple[str, int], list[tuple[bytes, tuple[str, Any]]]] = defaultdict(list)
    for row in rows:
        dataset = str(row["dataset"])
        label = int(row["label"])
        index = row["index"]
        digest = hashlib.sha256(
            f"{seed}\0{dataset}\0{label}\0{index}".encode("utf-8")
        ).digest()
        strata[(dataset, label)].append((digest, (dataset, index)))
    selected: set[tuple[str, Any]] = set()
    for candidates in strata.values():
        count = max(1, int(len(candidates) * fraction + 0.5))
        selected.update(key for _, key in sorted(candidates)[:count])
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def write_holdout_split(
    output_dir: Path,
    rows: list[dict[str, Any]],
    train_keys: set[tuple[str, Any]],
) -> int:
    labels_dir = output_dir / "holdout_splits" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (str(row["dataset"]), row["index"]) not in train_keys:
            by_dataset[str(row["dataset"])].append(row)
    datasets = []
    count = 0
    for dataset, group in sorted(by_dataset.items()):
        label_path = labels_dir / (dataset.replace("/", "__") + ".csv")
        with label_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "deceptive"])
            writer.writeheader()
            for row in sorted(group, key=lambda item: str(item["index"])):
                writer.writerow({
                    "index": row["index"],
                    "deceptive": bool(row["label"]),
                })
        count += len(group)
        datasets.append({
            "name": dataset,
            "labels_uri": label_path.resolve().as_posix(),
            "id_column": "index",
            "label_column": "deceptive",
        })
    split_path = output_dir / "holdout_splits" / "dry.train.yaml"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(yaml.safe_dump({"datasets": datasets}, sort_keys=False))
    return count


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def build(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    base_rows = [
        row
        for row in load_jsonl(args.base_teacher)
        if not row.get("parse_error")
        and row.get("label_match")
        and "varied-deception" in str(row.get("dataset", ""))
    ]
    if len(base_rows) != args.expected_base_rows:
        raise ValueError(
            f"expected {args.expected_base_rows} base rows, got {len(base_rows)}"
        )
    base = keyed(base_rows, ("dataset", "index"))
    generations = keyed(
        load_jsonl(args.generations),
        ("dataset", "index", "ensemble_member"),
    )
    distributions = keyed(
        load_jsonl(args.direct_distributions),
        ("dataset", "index", "ensemble_member"),
    )
    config = yaml.safe_load(args.generation_config.read_text())
    prompts = {
        str(member["name"]): str(member["prompt"])
        for member in config["ensemble"]["members"]
    }
    if set(prompts) != set(MEMBER_MAP):
        raise ValueError(f"unexpected generated members: {sorted(prompts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    complete_rows: list[dict[str, Any]] = []
    target_tokens: list[int] = []
    source_rating_counts: Counter[int] = Counter()
    source_label_matches = 0
    failures: list[dict[str, Any]] = []

    for (dataset, index), base_row in base.items():
        member_rows = []
        for generated_member, direct_member in MEMBER_MAP.items():
            generation = generations.get((dataset, index, generated_member))
            distribution = distributions.get((dataset, index, direct_member))
            error = None
            if generation is None or distribution is None:
                error = "missing_member_artifact"
            elif generation.get("parse_error"):
                error = "generation_parse_error"
            elif distribution.get("parse_error"):
                error = "distribution_parse_error"
            elif distribution.get("missing_rating_token_ids"):
                error = "missing_rating_logits"
            elif "</think>" not in str(generation.get("text", "")):
                error = "unclosed_thinking"
            if error is not None:
                failures.append({
                    "dataset": dataset,
                    "index": index,
                    "member": generated_member,
                    "error": error,
                })
                member_rows = []
                break
            probabilities = {
                str(rating): float(distribution["rating_probs"][str(rating)])
                for rating in range(1, 8)
            }
            if abs(sum(probabilities.values()) - 1.0) > 1e-9:
                raise ValueError(
                    f"rating distribution does not normalize for "
                    f"{(dataset, index, generated_member)}"
                )
            target = str(generation["text"])
            token_count = len(tokenizer.encode(target, add_special_tokens=False))
            if token_count > args.max_target_tokens:
                failures.append({
                    "dataset": dataset,
                    "index": index,
                    "member": generated_member,
                    "error": f"target_tokens={token_count}",
                })
                member_rows = []
                break
            rating = int(generation["rating"])
            member_rows.append({
                **base_row,
                "student_prompt": (
                    prompts[generated_member].rstrip()
                    + context_suffix(str(base_row["student_prompt"]))
                ),
                "student_target": target,
                "teacher_model": "Qwen/Qwen3.5-27B",
                "teacher_prompt_family": generated_member,
                "ensemble_member": generated_member,
                "direct_teacher_member": direct_member,
                "soft_rating_probs": probabilities,
                "source_rating": rating,
                "source_generation_sha256": hashlib.sha256(
                    target.encode("utf-8")
                ).hexdigest(),
                "parse_error": False,
                "label_match": int(rating > 1) == int(base_row["label"]),
                "prediction_source": "qwen27_generated_rating",
                "_target_tokens": token_count,
            })
        if len(member_rows) == 3:
            for row in member_rows:
                target_tokens.append(int(row.pop("_target_tokens")))
                source_rating_counts[int(row["source_rating"])] += 1
                source_label_matches += int(bool(row["label_match"]))
            complete_rows.extend(member_rows)

    complete_keys = {
        (str(row["dataset"]), row["index"]) for row in complete_rows
    }
    if len(complete_keys) < args.min_complete_rows:
        raise ValueError(
            f"complete row coverage {len(complete_keys)} is below "
            f"{args.min_complete_rows}"
        )
    if len(complete_rows) != 3 * len(complete_keys):
        raise AssertionError("complete rows do not have exactly three members")

    row_level = [
        base[key] for key in sorted(complete_keys, key=lambda item: (item[0], str(item[1])))
    ]
    train_keys = stable_train_keys(row_level, args.train_fraction, args.seed)
    train_records = [
        row
        for row in complete_rows
        if (str(row["dataset"]), row["index"]) in train_keys
    ]
    output_dir = args.output_dir
    write_jsonl(output_dir / "teacher" / "train.jsonl", train_records)
    holdout_count = write_holdout_split(output_dir, row_level, train_keys)

    audit = {
        "base_rows": len(base_rows),
        "complete_rows": len(complete_keys),
        "train_rows": len(train_keys),
        "train_member_targets": len(train_records),
        "holdout_rows": holdout_count,
        "failures": failures,
        "source_rating_counts": dict(sorted(source_rating_counts.items())),
        "source_rating_label_match_fraction": (
            source_label_matches / len(complete_rows)
        ),
        "target_tokens": {
            "min": min(target_tokens),
            "median": percentile(target_tokens, 0.5),
            "p95": percentile(target_tokens, 0.95),
            "max": max(target_tokens),
        },
        "train_fraction": args.train_fraction,
        "seed": args.seed,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--direct-distributions", type=Path, required=True)
    parser.add_argument("--base-teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--expected-base-rows", type=int, default=2877)
    parser.add_argument("--min-complete-rows", type=int, default=2850)
    parser.add_argument("--max-target-tokens", type=int, default=4096)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
