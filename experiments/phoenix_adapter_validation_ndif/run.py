#!/usr/bin/env python3
"""Re-evaluate migrated Phoenix adapters on validation through NDIF."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_NDIF_HOST = "https://aletheias.api.ndif.us"
MAX_PROMPT_TOKENS = 2048
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/blackbox/phoenix_adapter_validation_ndif_v1"
)


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    repo_id: str | None
    training_prompt: str


ADAPTERS = (
    AdapterSpec("base_qwen", None, "summary"),
    AdapterSpec(
        "gptoss_pi",
        "Jazhyc/aletheias-phoenix-wright-v2-1-adapter",
        "summary",
    ),
    AdapterSpec(
        "gptoss_blind",
        "Jazhyc/aletheias-phoenix-blind-reasoning-r16",
        "summary",
    ),
    AdapterSpec(
        "luna_pi",
        "Jazhyc/aletheias-phoenix-luna-medium-reasoning-r16",
        "summary",
    ),
    AdapterSpec(
        "qwen27_soft",
        "Jazhyc/aletheias-phoenix-v5-qwen27-tvg-soft-r16",
        "binary",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=ROOT / "dev_splits",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--adapter",
        action="append",
        choices=[spec.name for spec in ADAPTERS],
        help="repeat to score a subset; defaults to base plus all four adapters",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        choices=("summary", "binary"),
        help="repeat to score a subset; defaults to both prompt contracts",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--remote-batches-per-session", type=int, default=0)
    return parser.parse_args()


def load_credentials() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    os.environ.setdefault("NDIF_HOST", DEFAULT_NDIF_HOST)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is missing")


def prompt_templates() -> dict[str, str]:
    summary = yaml.safe_load(
        (ROOT / "configs/privileged_information_distillation.yaml").read_text()
    )["student"]["prompt"]
    binary = yaml.safe_load(
        (
            ROOT
            / "configs/pid_qwen27_tvg_binary_soft_distillation_v1.yaml"
        ).read_text()
    )["student"]["prompt"]
    if summary == binary:
        raise ValueError("summary and binary prompt contracts unexpectedly match")
    return {"summary": str(summary), "binary": str(binary)}


def load_records(splits_dir: Path, limit: int | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    manifest = yaml.safe_load((splits_dir / "dry.validation.yaml").read_text())
    rows: list[dict[str, Any]] = []
    for declaration in manifest["datasets"]:
        dataset_name = str(declaration["name"])
        labels_path = Path(str(declaration["labels_uri"]))
        if not labels_path.is_absolute():
            labels_path = splits_dir.parent / labels_path
        labels = pd.read_csv(labels_path)
        label_by_index = dict(
            zip(
                labels[str(declaration["id_column"])],
                labels[str(declaration["label_column"])].astype(int),
                strict=True,
            )
        )
        dataset = load_dataset(dataset_name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        found: set[Any] = set()
        for row in dataset:
            index = row["index"]
            if index not in label_by_index:
                continue
            found.add(index)
            rows.append(
                {
                    "dataset": dataset_name,
                    "index": index,
                    "label": label_by_index[index],
                    "messages": row["messages"],
                }
            )
        if found != set(label_by_index):
            raise RuntimeError(
                f"{dataset_name}: found {len(found)}/{len(label_by_index)} "
                "validation rows"
            )
    frame = pd.DataFrame(rows)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    return frame.reset_index(drop=True)


def binary_token_ids(tokenizer: Any) -> list[int]:
    ids = [
        tokenizer.encode(label, add_special_tokens=False)
        for label in ("0", "1")
    ]
    if any(len(encoded) != 1 for encoded in ids):
        raise ValueError(f"binary labels are not single tokens: {ids}")
    result = [int(encoded[0]) for encoded in ids]
    if len(set(result)) != 2:
        raise ValueError(f"binary labels do not have distinct ids: {result}")
    return result


def training_member_prompt(messages: Any, prompt_template: str) -> str:
    from experiments.privileged_information_distillation.core import (
        build_student_prompt,
    )

    return build_student_prompt(
        messages,
        prompt_template,
        3000,
        "tail",
        include_reasoning=False,
    )


def phoenix6_member_prompt(messages: Any, prompt_template: str) -> str:
    from experiments.phoenix_renderer_caps.run import (
        CONDITIONS,
        build_member_prompt,
    )

    rendered, _ = build_member_prompt(messages, CONDITIONS[0])
    _, separator, evidence = rendered.partition("<context>")
    if not separator:
        raise ValueError("Phoenix 6.0 prompt is missing <context>")
    return f"{prompt_template}\n\n<context>{evidence}"


def render_prompts(
    frame: pd.DataFrame,
    tokenizer: Any,
    templates: dict[str, str],
    selected_prompts: list[str],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    rendered: dict[str, list[str]] = {}
    parity: dict[str, Any] = {}
    for name in selected_prompts:
        training_members = [
            training_member_prompt(messages, templates[name])
            for messages in frame["messages"]
        ]
        phoenix6_members = [
            phoenix6_member_prompt(messages, templates[name])
            for messages in frame["messages"]
        ]
        mismatches = [
            position
            for position, (left, right) in enumerate(
                zip(training_members, phoenix6_members, strict=True)
            )
            if left != right
        ]
        parity[name] = {
            "rows": len(frame),
            "byte_equal_rows": len(frame) - len(mismatches),
            "different_rows": len(mismatches),
            "first_different_keys": [
                {
                    "dataset": str(frame.iloc[position]["dataset"]),
                    "index": frame.iloc[position]["index"],
                }
                for position in mismatches[:10]
            ],
        }
        rendered[name] = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": member}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "Prediction:"
            for member in training_members
        ]
    return rendered, parity


def position_batches(lengths: list[int]) -> list[list[int]]:
    order = np.argsort(lengths)
    batches: list[list[int]] = []
    cursor = 0
    while cursor < len(order):
        cap = 48
        candidate = order[cursor : min(cursor + cap, len(order))]
        if max(lengths[position] for position in candidate) > 600:
            cap = min(cap, 32)
            candidate = order[cursor : min(cursor + cap, len(order))]
        if max(lengths[position] for position in candidate) > 900:
            cap = min(cap, 16)
            candidate = order[cursor : min(cursor + cap, len(order))]
        batches.append(candidate.tolist())
        cursor += len(candidate)
    return batches


def encode_batches(
    tokenizer: Any,
    prompts: list[str],
) -> tuple[list[tuple[list[int], Any]], dict[str, Any]]:
    lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    ]
    batches = position_batches(lengths)
    encoded = []
    for positions in batches:
        tokens = tokenizer(
            [prompts[position] for position in positions],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        )
        encoded.append((positions, tokens))
    return encoded, {
        "rows": len(prompts),
        "batches": len(encoded),
        "min_tokens": min(lengths),
        "median_tokens": float(np.median(lengths)),
        "p95_tokens": float(np.quantile(lengths, 0.95)),
        "max_tokens": max(lengths),
        "truncated_rows": sum(length > MAX_PROMPT_TOKENS for length in lengths),
        "batch_shapes": [
            [len(positions), int(tokens["input_ids"].shape[1])]
            for positions, tokens in encoded
        ],
    }


def score_adapter(
    spec: AdapterSpec,
    prompts: dict[str, list[str]],
    selected_prompts: list[str],
    *,
    remote_batches_per_session: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from nnsight import LanguageModel

    model = (
        LanguageModel(MODEL_ID)
        if spec.repo_id is None
        else LanguageModel(MODEL_ID, peft=spec.repo_id)
    )
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = binary_token_ids(tokenizer)

    batches_by_prompt: dict[str, list[tuple[list[int], Any]]] = {}
    prompt_audits = {}
    flat_batches: list[tuple[str, list[int], Any]] = []
    for prompt_name in selected_prompts:
        batches, audit = encode_batches(tokenizer, prompts[prompt_name])
        batches_by_prompt[prompt_name] = batches
        prompt_audits[prompt_name] = audit
        flat_batches.extend(
            (prompt_name, positions, encoded)
            for positions, encoded in batches
        )
    scores = {
        name: np.full(len(prompts[name]), np.nan, dtype=np.float64)
        for name in selected_prompts
    }
    group_size = (
        len(flat_batches)
        if remote_batches_per_session <= 0
        else remote_batches_per_session
    )
    started = time.perf_counter()
    for group_start in range(0, len(flat_batches), group_size):
        group = flat_batches[group_start : group_start + group_size]
        pieces = []
        print(
            f"adapter={spec.name} batches={group_start + 1}-"
            f"{group_start + len(group)}/{len(flat_batches)}",
            flush=True,
        )
        with model.session(remote=True):
            for _, _, encoded in group:
                with model.trace(
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "logits_to_keep": 1,
                    }
                ):
                    logits = model.output.logits[:, -1, label_ids].float()
                    piece = (
                        torch.softmax(logits, dim=-1)[:, 1]
                        .detach()
                        .cpu()
                    )
                    pieces.append(piece)
            group_scores = torch.cat(pieces, dim=0).save()
        group_values = np.asarray(
            group_scores.float().tolist(),
            dtype=np.float64,
        )
        cursor = 0
        for prompt_name, positions, _ in group:
            count = len(positions)
            scores[prompt_name][positions] = group_values[
                cursor : cursor + count
            ]
            cursor += count
    elapsed = time.perf_counter() - started

    for prompt_name in selected_prompts:
        if np.isnan(scores[prompt_name]).any():
            raise RuntimeError(
                f"{spec.name}/{prompt_name}: "
                f"{int(np.isnan(scores[prompt_name]).sum())} missing scores"
            )
    return scores, {
        "elapsed_seconds": elapsed,
        "label_token_ids": label_ids,
        "prompts": prompt_audits,
    }


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = scores >= threshold
    positive = labels == 1
    negative = ~positive
    recall = float(predictions[positive].mean())
    fpr = float(predictions[negative].mean())
    return {
        "rows": int(len(labels)),
        "auroc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float((recall + 1.0 - fpr) / 2.0),
        "recall": recall,
        "fpr": fpr,
        "unique_scores": int(np.unique(scores).size),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }


def condition_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> dict[str, Any]:
    evaluated = frame.drop(columns="messages").copy()
    evaluated["score"] = scores
    per_dataset = {
        dataset: binary_metrics(
            group["label"].to_numpy(),
            group["score"].to_numpy(),
        )
        for dataset, group in evaluated.groupby("dataset", sort=True)
    }
    scenario = {}
    for name in ("instructed", "varied"):
        subset = evaluated[
            evaluated["dataset"].str.contains(f"dev-{name}-deception")
        ]
        scenario[name] = {
            "datasets": int(subset["dataset"].nunique()),
            "macro_auroc": float(
                np.mean(
                    [
                        binary_metrics(
                            group["label"].to_numpy(),
                            group["score"].to_numpy(),
                        )["auroc"]
                        for _, group in subset.groupby("dataset", sort=True)
                    ]
                )
            ),
        }
    all_metrics = binary_metrics(
        evaluated["label"].to_numpy(),
        evaluated["score"].to_numpy(),
    )
    all_metrics["macro_auroc"] = float(
        np.mean([metrics["auroc"] for metrics in per_dataset.values()])
    )
    return {
        "all": all_metrics,
        "scenario": scenario,
        "per_dataset": per_dataset,
    }


def paired_report(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    delta = right - left
    return {
        "exact_equal": int(np.sum(left == right)),
        "close_at_1e_6": int(
            np.sum(np.isclose(left, right, atol=1e-6, rtol=0.0))
        ),
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "mean_absolute_difference": float(np.mean(np.abs(delta))),
        "max_absolute_difference": float(np.max(np.abs(delta))),
    }


def hosted_metadata(repo_id: str | None) -> dict[str, Any]:
    if repo_id is None:
        return {
            "repo_id": None,
            "revision": None,
            "weight_sha256": None,
            "weight_bytes": None,
        }
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    weights = next(
        sibling
        for sibling in info.siblings
        if sibling.rfilename == "adapter_model.safetensors"
    )
    lfs = getattr(weights, "lfs", None)
    return {
        "repo_id": repo_id,
        "revision": info.sha,
        "weight_sha256": lfs.sha256 if lfs is not None else None,
        "weight_bytes": weights.size,
    }


def main() -> None:
    args = parse_args()
    load_credentials()
    selected_names = args.adapter or [spec.name for spec in ADAPTERS]
    selected_specs = [
        spec for spec in ADAPTERS if spec.name in selected_names
    ]
    selected_prompts = args.prompt or ["summary", "binary"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.splits_dir.resolve(), args.limit)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts, renderer_parity = render_prompts(
        records,
        tokenizer,
        prompt_templates(),
        selected_prompts,
    )
    prompt_hashes = {
        name: hashlib.sha256(
            "\0".join(values).encode("utf-8")
        ).hexdigest()
        for name, values in prompts.items()
    }

    all_scores: dict[str, dict[str, np.ndarray]] = {}
    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "split": "validation",
        "rows": len(records),
        "datasets": int(records["dataset"].nunique()),
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "selected_prompts": selected_prompts,
        "renderer_parity": renderer_parity,
        "prompt_set_sha256": prompt_hashes,
        "adapters": {},
    }
    for spec in selected_specs:
        metadata = hosted_metadata(spec.repo_id)
        print(
            f"starting adapter={spec.name} repo={spec.repo_id} "
            f"revision={metadata['revision']}",
            flush=True,
        )
        scores_by_prompt, timing = score_adapter(
            spec,
            prompts,
            selected_prompts,
            remote_batches_per_session=args.remote_batches_per_session,
        )
        all_scores[spec.name] = scores_by_prompt
        adapter_report = {
            "spec": asdict(spec),
            "hosted": metadata,
            "timing": timing,
            "conditions": {
                prompt_name: condition_metrics(records, scores)
                for prompt_name, scores in scores_by_prompt.items()
            },
            "prompt_sensitivity": (
                paired_report(
                    scores_by_prompt["summary"],
                    scores_by_prompt["binary"],
                )
                if {"summary", "binary"} <= scores_by_prompt.keys()
                else None
            ),
        }
        report["adapters"][spec.name] = adapter_report
        prediction_frame = records.drop(columns="messages").copy()
        for prompt_name, scores in scores_by_prompt.items():
            prediction_frame[f"score_{prompt_name}"] = scores
        prediction_frame.to_csv(
            output_dir / f"{spec.name}_predictions.csv",
            index=False,
        )
        (output_dir / "report.partial.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        matched = adapter_report["conditions"][spec.training_prompt]["all"]
        print(
            f"finished adapter={spec.name} "
            f"training_prompt={spec.training_prompt} "
            f"macro_auroc={matched['macro_auroc']:.6f} "
            f"unique={matched['unique_scores']}",
            flush=True,
        )

    if "base_qwen" in all_scores:
        base = all_scores["base_qwen"]
        for spec in selected_specs:
            if spec.name == "base_qwen":
                continue
            report["adapters"][spec.name]["activation_vs_base"] = {
                prompt_name: paired_report(
                    base[prompt_name],
                    all_scores[spec.name][prompt_name],
                )
                for prompt_name in selected_prompts
            }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    partial = output_dir / "report.partial.json"
    if partial.exists():
        partial.unlink()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
