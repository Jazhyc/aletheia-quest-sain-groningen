#!/usr/bin/env python3
"""Compare Phoenix 4.0 direct margins across vLLM and NDIF on one organism."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_submissions.phoenix_wright_v3 import build_member_prompt


MODEL_ID = "Qwen/Qwen3.5-9B"
ADAPTER_REPO = "Jazhyc/aletheias-phoenix-wright-v2-adapter"
LEGACY_ADAPTER_REVISION = "cb1d515230fc2f6615287ca75ee4650059b0ea28"
DATASET_NAME = "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None"
MAX_PROMPT_TOKENS = 2048
OUTPUT_ROOT = ROOT / "results/blackbox/phoenix_backend_parity_eunomia_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("vllm", "ndif"), required=True)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    return parser.parse_args()


def load_credentials() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    os.environ.setdefault("NDIF_HOST", "https://aletheias.api.ndif.us")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def dataset_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    matches = [
        config
        for config in payload["datasets"]
        if config["name"] == DATASET_NAME
    ]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one config for {DATASET_NAME}")
    return matches[0]


def combined_labels(splits_dir: Path) -> dict[Any, int]:
    labels: dict[Any, int] = {}
    for split in ("train", "validation", "test"):
        config = dataset_config(splits_dir / f"dry.{split}.yaml")
        frame = pd.read_csv(ROOT / config["labels_uri"])
        for index, label in zip(
            frame[config["id_column"]],
            frame[config["label_column"]],
            strict=True,
        ):
            if index in labels:
                raise ValueError(f"duplicate label for index={index}")
            labels[index] = int(label)
    if len(labels) != 400:
        raise ValueError(f"expected 400 combined labels, found {len(labels)}")
    return labels


def load_records(
    splits_dir: Path,
    *,
    limit_per_dataset: int | None,
) -> pd.DataFrame:
    from datasets import load_dataset

    labels = combined_labels(splits_dir)
    dataset = load_dataset(DATASET_NAME, split="test")
    if "index" not in dataset.column_names:
        dataset = dataset.add_column("index", list(range(len(dataset))))
    rows = [
        {
            "dataset": DATASET_NAME,
            "index": row["index"],
            "label": labels[row["index"]],
            "messages": row["messages"],
        }
        for row in dataset
        if row["index"] in labels
    ]
    rows.sort(key=lambda item: item["index"])
    if len(rows) != 400:
        raise ValueError(f"expected 400 fully labelled rows, found {len(rows)}")
    if limit_per_dataset is not None:
        rows = rows[:limit_per_dataset]
    return pd.DataFrame(rows)


def binary_token_ids(tokenizer: Any) -> list[int]:
    ids = [
        tokenizer.encode(label, add_special_tokens=False)
        for label in ("0", "1")
    ]
    if any(len(encoded) != 1 for encoded in ids):
        raise ValueError(f"binary labels are not single tokens: {ids}")
    return [int(encoded[0]) for encoded in ids]


def render_prompt(messages: Any, tokenizer: Any) -> str:
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_member_prompt(messages)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return chat + "Prediction:"


def prepare_prompts(records: pd.DataFrame, tokenizer: Any) -> pd.DataFrame:
    frame = records.copy()
    prompts = [
        render_prompt(messages, tokenizer)
        for messages in frame["messages"]
    ]
    token_ids = [
        tokenizer.encode(prompt, add_special_tokens=False)
        for prompt in prompts
    ]
    frame["prompt"] = prompts
    frame["prompt_tokens_untruncated"] = [len(ids) for ids in token_ids]
    frame["input_token_ids"] = [ids[-MAX_PROMPT_TOKENS:] for ids in token_ids]
    frame["prompt_sha256"] = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for prompt in prompts
    ]
    return frame


def logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def score_from_logprobs(values: dict[Any, Any], label_ids: list[int]) -> float:
    expanded = {int(key): logprob_value(value) for key, value in values.items()}
    missing = [token_id for token_id in label_ids if token_id not in expanded]
    if missing:
        raise ValueError(f"vLLM omitted requested label token ids: {missing}")
    margin = expanded[label_ids[1]] - expanded[label_ids[0]]
    return 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, margin))))


def generate_vllm_scores(
    llm: Any,
    prompts: list[Any],
    sampling: Any,
    label_ids: list[int],
    request: Any | None,
) -> np.ndarray:
    outputs = llm.generate(prompts, sampling, lora_request=request)
    scores = []
    for output in outputs:
        if not output.outputs or not output.outputs[0].logprobs:
            raise RuntimeError("vLLM returned no first-token logprobs")
        scores.append(
            score_from_logprobs(output.outputs[0].logprobs[0] or {}, label_ids)
        )
    return np.asarray(scores, dtype=np.float64)


def score_vllm(
    frame: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    legacy_adapter_dir = snapshot_download(
        ADAPTER_REPO,
        revision=LEGACY_ADAPTER_REVISION,
    )
    migrated_adapter_dir = snapshot_download(ADAPTER_REPO)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    label_ids = binary_token_ids(tokenizer)
    llm = LLM(
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=4096,
        language_model_only=True,
        skip_mm_profiling=True,
    )
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(label_ids),
        logprob_token_ids=label_ids,
        allowed_token_ids=label_ids,
    )
    prompts = [
        TokensPrompt(prompt_token_ids=ids)
        for ids in frame["input_token_ids"]
    ]
    requests = {
        "base": None,
        "legacy_adapter": LoRARequest(
            "phoenix-v2-legacy",
            1,
            legacy_adapter_dir,
        ),
        "migrated_adapter": LoRARequest(
            "phoenix-v2-migrated",
            2,
            migrated_adapter_dir,
        ),
    }
    conditions: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    for name, request in requests.items():
        print(f"vLLM condition={name}", flush=True)
        try:
            conditions[name] = generate_vllm_scores(
                llm,
                prompts,
                sampling,
                label_ids,
                request,
            )
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
            print(f"vLLM condition={name} failed: {errors[name]}", flush=True)
    if "legacy_adapter" not in conditions:
        raise RuntimeError(
            f"legacy vLLM adapter failed: {errors.get('legacy_adapter')}"
        )
    return conditions, errors


def position_batches(lengths: list[int]) -> list[list[int]]:
    order = np.argsort(lengths)
    batches: list[list[int]] = []
    cursor = 0
    while cursor < len(order):
        cap = 48
        candidate = order[cursor:min(cursor + cap, len(order))]
        if max(lengths[position] for position in candidate) > 600:
            cap = min(cap, 32)
            candidate = order[cursor:min(cursor + cap, len(order))]
        if max(lengths[position] for position in candidate) > 900:
            cap = min(cap, 16)
            candidate = order[cursor:min(cursor + cap, len(order))]
        batches.append(candidate.tolist())
        cursor += len(candidate)
    return batches


def score_ndif(frame: pd.DataFrame) -> np.ndarray:
    import torch
    from nnsight import LanguageModel

    if not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is missing")
    model = LanguageModel(MODEL_ID, peft=ADAPTER_REPO)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = binary_token_ids(tokenizer)
    scores = np.full(len(frame), np.nan, dtype=np.float64)

    for dataset_name, unit in frame.groupby("dataset", sort=True):
        positions = unit.index.to_list()
        prompts = unit["prompt"].tolist()
        lengths = unit["prompt_tokens_untruncated"].astype(int).tolist()
        batches = position_batches(lengths)
        encoded_batches = []
        for relative_positions in batches:
            encoded = tokenizer(
                [prompts[position] for position in relative_positions],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_PROMPT_TOKENS,
            )
            encoded_batches.append((relative_positions, encoded))
        print(
            f"NDIF unit={dataset_name} rows={len(unit)} batches={len(batches)}",
            flush=True,
        )
        pieces = []
        with model.session(remote=True):
            for relative_positions, encoded in encoded_batches:
                with model.trace({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "logits_to_keep": 1,
                }):
                    logits = model.output.logits[:, -1, label_ids].float()
                    piece = torch.softmax(logits, dim=-1)[:, 1].detach().cpu()
                    pieces.append(piece)
            saved_unit_scores = torch.cat(pieces).save()
        unit_values = np.asarray(
            saved_unit_scores.float().tolist(),
            dtype=np.float64,
        )
        offset = 0
        for relative_positions in batches:
            values = unit_values[offset:offset + len(relative_positions)]
            offset += len(relative_positions)
            for relative_position, value in zip(
                relative_positions, values, strict=True
            ):
                scores[positions[relative_position]] = value
    if np.isnan(scores).any():
        raise RuntimeError(f"NDIF left {int(np.isnan(scores).sum())} scores missing")
    return scores


def metric_report(frame: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    return {
        "rows": int(len(frame)),
        "dataset": DATASET_NAME,
        "auroc": float(roc_auc_score(frame["label"], frame["score"])),
        "score_min": float(frame["score"].min()),
        "score_median": float(frame["score"].median()),
        "score_max": float(frame["score"].max()),
        "fraction_at_or_above_0_15": float((frame["score"] >= 0.15).mean()),
        "unique_scores": int(frame["score"].nunique()),
    }


def paired_score_report(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    delta = right - left
    return {
        "exact_equal": int(np.sum(left == right)),
        "close_at_1e_6": int(np.sum(np.isclose(left, right, atol=1e-6, rtol=0))),
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "mean_absolute_difference": float(np.mean(np.abs(delta))),
        "max_absolute_difference": float(np.max(np.abs(delta))),
    }


def main() -> None:
    args = parse_args()
    load_credentials()
    output_dir = (args.output_dir or OUTPUT_ROOT / args.backend).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    records = load_records(
        args.splits_dir.resolve(),
        limit_per_dataset=args.limit_per_dataset,
    )
    frame = prepare_prompts(records, tokenizer)
    started = time.perf_counter()
    vllm_conditions: dict[str, np.ndarray] = {}
    vllm_errors: dict[str, str] = {}
    if args.backend == "vllm":
        vllm_conditions, vllm_errors = score_vllm(frame, args)
        scores = vllm_conditions["migrated_adapter"]
    else:
        scores = score_ndif(frame)
    elapsed = time.perf_counter() - started
    frame["score"] = scores
    report = {
        "backend": args.backend,
        "model_id": MODEL_ID,
        "adapter_repo": ADAPTER_REPO,
        "adapter_revision": "main",
        "renderer": "phoenix_wright_v4_0",
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "elapsed_seconds": elapsed,
        "prompt_set_sha256": hashlib.sha256(
            "\0".join(frame["prompt_sha256"]).encode("ascii")
        ).hexdigest(),
        "metrics": metric_report(frame),
    }
    if args.backend == "vllm":
        report["vllm_errors"] = vllm_errors
        report["vllm_conditions"] = {}
        for name, condition_scores in vllm_conditions.items():
            frame[f"score_{name}"] = condition_scores
            condition_frame = frame.copy()
            condition_frame["score"] = condition_scores
            report["vllm_conditions"][name] = metric_report(condition_frame)
        base_scores = vllm_conditions.get("base")
        if base_scores is not None:
            report["vllm_activation_checks"] = {
                name: paired_score_report(base_scores, condition_scores)
                for name, condition_scores in vllm_conditions.items()
                if name != "base"
            }
    frame.drop(columns=["messages", "prompt", "input_token_ids"]).to_csv(
        output_dir / "predictions.csv",
        index=False,
    )
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
