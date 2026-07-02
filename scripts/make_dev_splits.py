#!/usr/bin/env python3
"""Create deterministic stratified train/validation/test splits for dev data.

The competition dev datasets are separate HF repos with a single ``test`` split.
This script creates local label subsets and dry-run YAML files so we can tune on
one subset and keep a held-out local test subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)
HF_SPLIT = "test"
BASE_MODELS = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
}
SPLIT_PREFIXES = ("dev-test-", "validation-", "dev-")


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    labels_uri: str
    id_column: str = "index"
    label_column: str = "deceptive"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-yaml", default="dry.yaml",
                   help="Source dry-run YAML listing public dev datasets.")
    p.add_argument("--out-dir", default="dev_splits",
                   help="Directory for split manifests, labels, and dry YAMLs.")
    p.add_argument("--seed", type=int, default=0,
                   help="Deterministic seed for row assignment.")
    p.add_argument("--ratios", type=float, nargs=3, default=DEFAULT_RATIOS,
                   metavar=("TRAIN", "VALIDATION", "TEST"),
                   help="Split ratios; default: 0.8 0.1 0.1.")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="Optional Hugging Face token for loading labels.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing output directory.")
    return p.parse_args()


def safe_name(dataset: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", dataset)


def parse_dataset_name(name: str) -> tuple[str, str, str | None, str | None]:
    tail = name.split("/")[-1]
    source_split = ""
    for prefix in SPLIT_PREFIXES:
        if tail.startswith(prefix):
            source_split = prefix.rstrip("-")
            tail = tail[len(prefix):]
            break

    best: tuple[int, str, str] | None = None
    for token, model_id in BASE_MODELS.items():
        idx = tail.find(token)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, token, model_id)
    if best is None:
        return source_split, tail.strip("-") or tail, None, None

    idx, token, model_id = best
    scenario = tail[:idx].strip("-")
    rest = tail[idx + len(token):].strip("-")
    lora_id = rest if rest and rest.lower() != "none" else None
    return source_split, scenario or tail, model_id, lora_id


def load_entries(path: Path) -> tuple[dict, list[DatasetEntry]]:
    data = yaml.safe_load(path.read_text()) or {}
    entries = [DatasetEntry(**d) for d in data.get("datasets", [])]
    if not entries:
        raise SystemExit(f"{path} has no datasets entries")
    return data, entries


def load_labels(entry: DatasetEntry, token: str | None) -> list[dict]:
    path = Path(entry.labels_uri)
    if entry.labels_uri.endswith(".csv") and path.exists():
        import pandas as pd

        raw = pd.read_csv(path)
        records = raw.to_dict("records")
    else:
        from datasets import load_dataset

        ds = load_dataset(entry.labels_uri, split=HF_SPLIT, token=token)
        records = list(ds)

    if not records:
        return []
    if entry.label_column not in records[0]:
        raise SystemExit(f"{entry.labels_uri} missing label column {entry.label_column!r}")
    return records


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "::".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    child_seed = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(child_seed)


def allocate_counts(n: int, ratios: tuple[float, float, float]) -> list[int]:
    if n <= 0:
        return [0, 0, 0]
    raw = np.array(ratios, dtype=float) * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    order = sorted(range(len(ratios)), key=lambda i: (raw[i] - counts[i], ratios[i]), reverse=True)
    for i in order[:remainder]:
        counts[i] += 1

    # For small but splittable strata, keep every requested split represented.
    if n >= len(ratios):
        for i, ratio in enumerate(ratios):
            if ratio <= 0 or counts[i] > 0:
                continue
            donor = max((j for j in range(len(ratios)) if counts[j] > 1),
                        key=lambda j: counts[j], default=None)
            if donor is not None:
                counts[donor] -= 1
                counts[i] += 1
    return counts.tolist()


def assign_group(rows: list[dict], seed: int, key: tuple, ratios: tuple[float, float, float]) -> list[tuple[str, dict]]:
    order = np.arange(len(rows))
    stable_rng(seed, *key).shuffle(order)
    counts = allocate_counts(len(rows), ratios)
    out: list[tuple[str, dict]] = []
    offset = 0
    for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
        for pos in order[offset:offset + count]:
            out.append((split_name, rows[int(pos)]))
        offset += count
    return out


def normalize_ratios(values: Iterable[float]) -> tuple[float, float, float]:
    ratios = tuple(float(v) for v in values)
    total = sum(ratios)
    if len(ratios) != 3 or total <= 0 or any(v < 0 for v in ratios):
        raise SystemExit("--ratios must be three non-negative values with positive sum")
    return tuple(v / total for v in ratios)  # type: ignore[return-value]


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dry_data, entries = load_entries(Path(args.dry_yaml))
    ratios = normalize_ratios(args.ratios)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{out_dir} already exists; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    per_split_labels: dict[str, dict[str, list[dict]]] = {
        split: defaultdict(list) for split in SPLIT_NAMES
    }

    for entry in entries:
        source_split, scenario, model_id, lora_id = parse_dataset_name(entry.name)
        rows = load_labels(entry, args.hf_token)
        grouped: dict[bool, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[bool(row[entry.label_column])].append(row)

        for label, group_rows in sorted(grouped.items()):
            assigned = assign_group(
                group_rows, args.seed, (entry.name, entry.label_column, label), ratios
            )
            for dev_split, row in assigned:
                idx = row.get(entry.id_column)
                if idx is None:
                    raise SystemExit(f"{entry.labels_uri} missing id column {entry.id_column!r}")
                label_value = bool(row[entry.label_column])
                manifest_rows.append({
                    "dataset": entry.name,
                    "index": idx,
                    "deceptive": label_value,
                    "dev_split": dev_split,
                    "source_split": source_split,
                    "scenario": scenario,
                    "model_id": model_id or "",
                    "lora_id": lora_id or "",
                })
                per_split_labels[dev_split][entry.name].append({
                    "index": idx,
                    "deceptive": label_value,
                })

    manifest_rows.sort(key=lambda r: (r["dataset"], r["dev_split"], int(r["deceptive"]), str(r["index"])))
    write_csv(
        out_dir / "manifest.csv",
        manifest_rows,
        ["dataset", "index", "deceptive", "dev_split", "source_split", "scenario", "model_id", "lora_id"],
    )

    for split_name in SPLIT_NAMES:
        split_entries = []
        for entry in entries:
            label_rows = sorted(per_split_labels[split_name][entry.name],
                                key=lambda r: str(r["index"]))
            label_path = out_dir / "labels" / split_name / f"{safe_name(entry.name)}.csv"
            write_csv(label_path, label_rows, ["index", "deceptive"])
            split_entries.append({
                "name": entry.name,
                "labels_uri": label_path.as_posix(),
                "id_column": "index",
                "label_column": "deceptive",
            })
        split_yaml = {
            k: v for k, v in dry_data.items()
            if k not in {"datasets"}
        }
        split_yaml["datasets"] = split_entries
        (out_dir / f"dry.{split_name}.yaml").write_text(
            yaml.safe_dump(split_yaml, sort_keys=False)
        )

    print(f"Wrote {out_dir / 'manifest.csv'}")
    print("Split summary:")
    counts = Counter((r["dev_split"], r["deceptive"]) for r in manifest_rows)
    totals = Counter(r["dev_split"] for r in manifest_rows)
    for split_name in SPLIT_NAMES:
        honest = counts[(split_name, False)]
        deceptive = counts[(split_name, True)]
        print(f"  {split_name:10s} {totals[split_name]:6d} rows "
              f"({deceptive} deceptive / {honest} honest)")
    print(f"Wrote dry-run configs under {out_dir}/dry.<split>.yaml")


if __name__ == "__main__":
    main()
