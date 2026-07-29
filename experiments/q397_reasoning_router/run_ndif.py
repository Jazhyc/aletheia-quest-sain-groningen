#!/usr/bin/env python3
"""Score direct and post-reasoning Q397 margins through optimized NDIF sessions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_adapter_reasoning_ndif.run import (  # noqa: E402
    encode_generation_batches,
    generate_reasoning,
    render_generation_prompts,
    score_prompt_sets,
)
from experiments.phoenix_adapter_validation_ndif.run import (  # noqa: E402
    AdapterSpec,
    MAX_PROMPT_TOKENS,
    MODEL_ID,
    binary_token_ids,
    condition_metrics,
    hosted_metadata,
    load_credentials,
    paired_report,
    prompt_templates,
    training_member_prompt,
)
from submission.phoenix_wright import prefix_before_prediction  # noqa: E402


ADAPTER = AdapterSpec(
    "q397_optimized_ep2",
    "Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16-ep2",
    "binary",
)
MAX_NEW_TOKENS = 512
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/blackbox/q397_reasoning_router_validation_v1"
)
TEST_OUTPUT_DIR = ROOT / "results/blackbox/q397_reasoning_router_test_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generation-batches-per-session", type=int, default=0)
    parser.add_argument("--score-batches-per-session", type=int, default=0)
    return parser.parse_args()


def load_split_records(
    splits_dir: Path,
    split: str,
    limit: int | None = None,
) -> pd.DataFrame:
    from datasets import load_dataset

    manifest = yaml.safe_load((splits_dir / f"dry.{split}.yaml").read_text())
    rows: list[dict[str, Any]] = []
    for declaration in manifest["datasets"]:
        dataset_name = str(declaration["name"])
        labels_path = Path(str(declaration["labels_uri"]))
        if not labels_path.is_absolute():
            labels_path = splits_dir.parent / labels_path
        labels = pd.read_csv(labels_path)
        label_by_index = dict(zip(
            labels[str(declaration["id_column"])],
            labels[str(declaration["label_column"])].astype(int),
            strict=True,
        ))
        dataset = load_dataset(dataset_name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        found: set[Any] = set()
        for row in dataset:
            index = row["index"]
            if index not in label_by_index:
                continue
            found.add(index)
            rows.append({
                "dataset": dataset_name,
                "index": index,
                "label": label_by_index[index],
                "messages": row["messages"],
            })
        if found != set(label_by_index):
            raise RuntimeError(
                f"{dataset_name}: found {len(found)}/{len(label_by_index)} rows"
            )
    frame = pd.DataFrame(rows)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    return frame.reset_index(drop=True)


def summary_template() -> str:
    config = yaml.safe_load(
        (ROOT / "configs/privileged_information_distillation.yaml").read_text()
    )
    return str(config["student"]["prompt"])


def render_prompts(
    records: pd.DataFrame,
    tokenizer: Any,
) -> tuple[list[str], list[str]]:
    generation = render_generation_prompts(
        records,
        tokenizer,
        summary_template(),
    )
    binary = render_generation_prompts(
        records,
        tokenizer,
        prompt_templates()["binary"],
    )
    return generation, binary


def main() -> None:
    args = parse_args()
    load_credentials()
    output_dir = (
        args.output_dir
        or (DEFAULT_OUTPUT_DIR if args.split == "validation" else TEST_OUTPUT_DIR)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_split_records(
        args.splits_dir.resolve(),
        args.split,
        args.limit,
    )

    from nnsight import LanguageModel

    metadata = hosted_metadata(ADAPTER.repo_id)
    print(
        f"starting adapter={ADAPTER.name} revision={metadata['revision']}",
        flush=True,
    )
    model = LanguageModel(MODEL_ID, peft=ADAPTER.repo_id)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = binary_token_ids(tokenizer)

    generation_prompts, binary_prompts = render_prompts(records, tokenizer)
    replies, generation_audit = generate_reasoning(
        model,
        tokenizer,
        generation_prompts,
        batches_per_session=args.generation_batches_per_session,
    )
    prompt_sets = {
        "direct": [prompt + "Prediction:" for prompt in binary_prompts],
        "post_reasoning": [
            prompt + prefix_before_prediction(reply)
            for prompt, reply in zip(generation_prompts, replies, strict=True)
        ],
    }
    scores, score_audit = score_prompt_sets(
        model,
        tokenizer,
        label_ids,
        prompt_sets,
        batches_per_session=args.score_batches_per_session,
    )
    conditions = {
        name: condition_metrics(records, values)
        for name, values in scores.items()
    }
    _, generation_shape = encode_generation_batches(
        tokenizer,
        generation_prompts,
    )
    report = {
        "method": f"q397_reasoning_router_{args.split}_v1",
        "model_id": MODEL_ID,
        "split": args.split,
        "rows": len(records),
        "datasets": int(records["dataset"].nunique()),
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "adapter": asdict(ADAPTER),
        "hosted": metadata,
        "generation": generation_audit,
        "generation_shape": generation_shape,
        "scoring": score_audit,
        "conditions": conditions,
        "post_reasoning_vs_direct": paired_report(
            scores["direct"],
            scores["post_reasoning"],
        ),
        "macro_auroc_delta": (
            conditions["post_reasoning"]["all"]["macro_auroc"]
            - conditions["direct"]["all"]["macro_auroc"]
        ),
    }

    predictions = records.drop(columns="messages").copy()
    predictions["direct_score"] = scores["direct"]
    predictions["post_reasoning_score"] = scores["post_reasoning"]
    predictions["reasoning_reply"] = replies
    predictions["reasoning_tokens"] = [
        len(tokenizer.encode(reply, add_special_tokens=False))
        for reply in replies
    ]
    predictions.to_json(
        output_dir / "scores.jsonl",
        orient="records",
        lines=True,
        double_precision=15,
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
