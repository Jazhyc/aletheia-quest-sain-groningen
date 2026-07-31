#!/usr/bin/env python3
"""Export the Phoenix 8.1 Kimi supervision as an MIT-only HF dataset.

The public Parquet intentionally contains annotations and source references,
not copied source conversations or labels.  ``hf_dataset/render_prompts.py``
can materialize the exact prompts and labels from their attributed upstream
datasets when a user needs a train-ready local file.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ASSET_DIR = Path(__file__).with_name("hf_dataset")
RESULTS = ROOT / "results" / "blackbox"
COMPETITION_VARIED = (
    RESULTS
    / "kimi_k3_fireworks_nothink_tvg_binary_logit_v1"
    / "train"
)
COMPETITION_INSTRUCTED = (
    RESULTS
    / "kimi_k3_fireworks_nothink_tvg_binary_logit_instructed_v1"
    / "train"
)
LIARS = RESULTS / "kimi_k3_liars_semantic_soft_v1" / "train"
COMBINED = RESULTS / "kimi_k3_tvg_soft_full_plus_liars_v1" / "train"
LIARS_REPO = "Cadenza-Labs/liars-bench"
# Byte-exact source snapshot used by the frozen selection in ``run.sh``.
LIARS_REVISION = "65299c5b10aa07adf75716ecb875c6713eed0dde"
TEACHER_MODEL = "moonshotai/kimi-k3"
EXPECTED_ROWS = 13_149


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hf-hub-cache",
        type=Path,
        default=None,
        help="HF Hub cache used to freeze upstream source revisions.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"artifact is empty: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def default_hub_cache() -> Path:
    if value := os.environ.get("HF_HUB_CACHE"):
        return Path(value)
    if value := os.environ.get("HF_HOME"):
        return Path(value) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cache_repo_dir(repo_id: str, hub_cache: Path) -> Path:
    return hub_cache / f"datasets--{repo_id.replace('/', '--')}"


def cached_revision(repo_id: str, hub_cache: Path) -> str:
    reference = cache_repo_dir(repo_id, hub_cache) / "refs" / "main"
    if not reference.is_file():
        raise FileNotFoundError(
            f"missing cached main revision for {repo_id}: {reference}"
        )
    revision = reference.read_text().strip()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError(f"invalid cached revision for {repo_id}: {revision!r}")
    return revision


def keyed(rows: list[dict[str, Any]], name: str) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["index"]))
        if key in result:
            raise ValueError(f"duplicate {name} identity: {key}")
        result[key] = row
    return result


def route_metadata(dataset: str, student: dict[str, Any]) -> dict[str, Any]:
    if not dataset.startswith("liars-bench/"):
        return {
            "source_collection": "aletheias_quest",
            "source_dataset": dataset,
            "source_config": "default",
            "source_index": str(student["index"]),
            "source_label_dataset": f"{dataset}-labels",
            "source_label_config": "default",
            "category": "competition",
            "student_prompt_kind": "ordinary",
            "teacher_prompt_kind": "ordinary_kimi_query",
            "student_context_chars": 3000,
            "student_context_truncation": "tail",
            "teacher_context_chars": 3000,
            "teacher_context_truncation": "tail",
        }

    category = dataset.removeprefix("liars-bench/")
    selected = str(student["teacher_prompt_kind"])
    student_settings = {
        "harm-pressure-choice": ("ordinary", 3000, "tail"),
        "harm-pressure-knowledge-report": ("epistemic", 6000, "head_tail"),
        "insider-trading": ("action", 4500, "head_tail"),
        "soft-trigger": ("ordinary", 3000, "tail"),
    }
    teacher_settings = {
        ("harm-pressure-choice", "semantic"): (
            "harm_pressure_choice",
            3000,
            "tail",
        ),
        ("harm-pressure-knowledge-report", "semantic"): (
            "epistemic",
            6000,
            "head_tail",
        ),
        ("insider-trading", "ordinary"): ("ordinary", 3000, "tail"),
        ("soft-trigger", "ordinary"): ("ordinary", 3000, "tail"),
    }
    try:
        student_kind, student_chars, student_truncation = student_settings[category]
        teacher_kind, teacher_chars, teacher_truncation = teacher_settings[
            (category, selected)
        ]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Liars route category={category!r} condition={selected!r}"
        ) from exc
    source_index = str(student["index"]).removeprefix(f"{category}:")
    return {
        "source_collection": "liars_bench",
        "source_dataset": LIARS_REPO,
        "source_config": category,
        "source_index": source_index,
        "source_label_dataset": LIARS_REPO,
        "source_label_config": category,
        "category": category,
        "student_prompt_kind": student_kind,
        "teacher_prompt_kind": teacher_kind,
        "student_context_chars": student_chars,
        "student_context_truncation": student_truncation,
        "teacher_context_chars": teacher_chars,
        "teacher_context_truncation": teacher_truncation,
    }


def build_rows(hub_cache: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    student_path = COMBINED / "student_rows.jsonl"
    soft_path = COMBINED / "soft_targets.jsonl"
    students = keyed(read_jsonl(student_path), "student")
    soft = keyed(read_jsonl(soft_path), "soft target")
    generations = keyed(
        read_jsonl(COMPETITION_VARIED / "generations.jsonl")
        + read_jsonl(COMPETITION_INSTRUCTED / "generations.jsonl")
        + read_jsonl(LIARS / "generations.jsonl"),
        "generation",
    )
    if not (set(students) == set(soft) == set(generations)):
        raise ValueError("student, soft-target, and generation identities differ")
    if len(students) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(students)}")

    source_repos = {
        metadata["source_dataset"]
        for key, student in students.items()
        for metadata in [route_metadata(key[0], student)]
    }
    source_repos.update(
        metadata["source_label_dataset"]
        for key, student in students.items()
        for metadata in [route_metadata(key[0], student)]
    )
    revisions = {
        repo_id: cached_revision(repo_id, hub_cache) for repo_id in sorted(source_repos)
    }
    revisions[LIARS_REPO] = LIARS_REVISION
    liars_snapshot = cache_repo_dir(LIARS_REPO, hub_cache) / "snapshots" / LIARS_REVISION
    if not liars_snapshot.is_dir():
        raise FileNotFoundError(f"frozen Liars snapshot is missing: {liars_snapshot}")

    exported: list[dict[str, Any]] = []
    for dataset, index in sorted(students):
        key = (dataset, index)
        student = students[key]
        target = soft[key]
        generation = generations[key]
        metadata = route_metadata(dataset, student)
        if int(student["label"]) != int(target["label"]) or int(student["label"]) != int(
            generation["label"]
        ):
            raise ValueError(f"source labels disagree for {key}")
        if str(generation["model"]) != TEACHER_MODEL:
            raise ValueError(f"unexpected teacher model for {key}: {generation['model']}")
        label_logprobs = {str(k): float(v) for k, v in generation["label_logprobs"].items()}
        probabilities = {
            str(k): float(v) for k, v in generation["target_probs"].items()
        }
        if set(label_logprobs) != {"0", "1"}:
            raise ValueError(f"missing literal label logprobs for {key}")
        if set(probabilities) != {"honest", "deceptive"}:
            raise ValueError(f"invalid normalized probabilities for {key}")
        denominator = math.exp(label_logprobs["0"]) + math.exp(label_logprobs["1"])
        expected_deceptive = math.exp(label_logprobs["1"]) / denominator
        if not math.isclose(expected_deceptive, probabilities["deceptive"], abs_tol=1e-12):
            raise ValueError(f"logprobs and probabilities disagree for {key}")
        if not math.isclose(float(target["soft_target"]), expected_deceptive, abs_tol=1e-12):
            raise ValueError(f"soft target and Kimi probability disagree for {key}")
        student_prompt = str(student["student_prompt"])
        exported.append(
            {
                "row_id": sha256_text(f"{dataset}\0{index}")[:24],
                "original_dataset_key": dataset,
                "original_index": index,
                **metadata,
                "source_revision": revisions[metadata["source_dataset"]],
                "source_label_revision": revisions[metadata["source_label_dataset"]],
                "source_split": "test",
                "teacher_model": TEACHER_MODEL,
                "teacher_provider": str(generation["provider"]),
                "teacher_reasoning_effort": "none",
                "teacher_logprob_0": label_logprobs["0"],
                "teacher_logprob_1": label_logprobs["1"],
                "teacher_probability_0": probabilities["honest"],
                "teacher_probability_1": probabilities["deceptive"],
                "teacher_log_odds_deceptive": label_logprobs["1"]
                - label_logprobs["0"],
                "soft_target_deceptive": float(target["soft_target"]),
                "teacher_prompt_sha256": str(generation["prompt_sha256"]),
                "student_prompt_sha256": sha256_text(student_prompt),
                "teacher_prompt_chars": int(generation["prompt_chars"]),
                "student_prompt_chars": len(student_prompt),
            }
        )
    return exported, revisions


def write_dataset(output_dir: Path, hub_cache: Path) -> dict[str, Any]:
    rows, revisions = build_rows(hub_cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    parquet = data_dir / "train-00000-of-00001.parquet"
    frame = pd.DataFrame(rows)
    frame.to_parquet(parquet, index=False, compression="zstd")

    for filename in ("README.md", "LICENSE", "render_prompts.py", "requirements.txt"):
        shutil.copy2(ASSET_DIR / filename, output_dir / filename)
    from experiments.kimi_liars_enrichment.prompts import (
        HARM_PRESSURE_CHOICE_BINARY_PROMPT,
    )
    from experiments.openrouter_qwen397_tvg.run_openrouter_tvg import (
        FROZEN_PROMPT_TEMPLATE,
    )
    from submission.phoenix_wright import ACTION_PROMPT, DECEPTION_PROMPT, EPISTEMIC_PROMPT

    templates = {
        "ordinary": DECEPTION_PROMPT,
        "ordinary_kimi_query": FROZEN_PROMPT_TEMPLATE,
        "epistemic": EPISTEMIC_PROMPT,
        "action": ACTION_PROMPT,
        "harm_pressure_choice": HARM_PRESSURE_CHOICE_BINARY_PROMPT,
    }
    (output_dir / "prompt_templates.json").write_text(
        json.dumps(templates, indent=2, sort_keys=True) + "\n"
    )

    counts = Counter(row["source_collection"] for row in rows)
    categories = Counter(row["category"] for row in rows)
    manifest = {
        "artifact": "Phoenix 8.1 Kimi K3 binary-logit distillation annotations",
        "rows": len(rows),
        "row_counts": dict(sorted(counts.items())),
        "category_counts": dict(sorted(categories.items())),
        "source_revisions": revisions,
        "teacher_model": TEACHER_MODEL,
        "teacher_score": "softmax over exact literal token logprobs for 0 and 1",
        "parquet_sha256": sha256_file(parquet),
        "prompt_template_sha256": {
            key: sha256_text(value) for key, value in sorted(templates.items())
        },
        "source_cache_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                COMPETITION_VARIED / "generations.jsonl",
                COMPETITION_INSTRUCTED / "generations.jsonl",
                LIARS / "generations.jsonl",
                COMBINED / "student_rows.jsonl",
                COMBINED / "soft_targets.jsonl",
            )
        },
        "license_scope": (
            "MIT covers these annotations and code. Upstream source text and labels are "
            "not redistributed and retain their original licenses."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    manifest["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
    return manifest


def main() -> None:
    args = parse_args()
    hub_cache = args.hf_hub_cache or default_hub_cache()
    manifest = write_dataset(args.output_dir.resolve(), hub_cache.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
