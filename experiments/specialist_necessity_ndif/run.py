#!/usr/bin/env python3
"""Benchmark Phoenix 6.3's optimized judge against its routed specialists."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import datetime as dt
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.evaluate_prompt_sweep import (  # noqa: E402
    split_records,
)
from experiments.liars_bench_distillation.prompts import (  # noqa: E402
    HARM_PRESSURE_CHOICE_PROMPT,
)
from experiments.phoenix_renderer_caps.run import (  # noqa: E402
    CONDITIONS as PHOENIX_CAP_CONDITIONS,
    build_member_prompt as build_structural_member_prompt,
)
from experiments.privileged_information_distillation.core import (  # noqa: E402
    build_student_prompt,
)
from submission.phoenix_wright import (  # noqa: E402
    EMPTY_REASONING_PREFIX,
    build_action_member_prompt,
    build_epistemic_member_prompt,
    build_intent_member_prompt,
    has_final_assistant_reasoning,
)


MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_NDIF_HOST = "https://aletheias.api.ndif.us"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/blackbox/specialist_necessity_ndif_v1"
)
DEFAULT_EXTERNAL_ARTIFACT = (
    ROOT / "results/blackbox/liars_bench_pid_aug_v1/eval.jsonl"
)
MAX_PROMPT_TOKENS = 2_048
REASONING_MAX_CHARS = 4_000
EPSILON = 1e-8


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    repo_id: str | None
    maximum_rank: int


@dataclass(frozen=True)
class Condition:
    name: str
    record_group: str
    adapter: str
    prompt_kind: str


ADAPTERS = {
    spec.name: spec
    for spec in (
        AdapterSpec(
            "q397_optimized",
            "Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16-ep2",
            16,
        ),
        AdapterSpec("base_qwen", None, 1),
        AdapterSpec(
            "current_action",
            "Jazhyc/aletheias-phoenix-wright-v2-adapter",
            16,
        ),
        AdapterSpec(
            "current_intent_rank1",
            "Jazhyc/aletheias-phoenix-v3-resolved-intent-r1",
            1,
        ),
    )
}


CONDITIONS = (
    Condition("action_q397_general_direct", "action", "q397_optimized", "general"),
    Condition("action_q397_action_direct", "action", "q397_optimized", "action_direct"),
    Condition("action_q397_action_empty", "action", "q397_optimized", "action_empty"),
    Condition(
        "action_current_specialist",
        "action",
        "current_action",
        "action_empty",
    ),
    Condition("hpkr_q397_general_direct", "hpkr", "q397_optimized", "general"),
    Condition("hpkr_q397_epistemic_direct", "hpkr", "q397_optimized", "hpkr_direct"),
    Condition("hpkr_q397_epistemic_empty", "hpkr", "q397_optimized", "hpkr_empty"),
    Condition(
        "hpkr_current_base_specialist",
        "hpkr",
        "base_qwen",
        "hpkr_empty",
    ),
    Condition(
        "harm_choice_q397_general_direct",
        "harm_choice",
        "q397_optimized",
        "general",
    ),
    Condition(
        "harm_choice_q397_specialist_direct",
        "harm_choice",
        "q397_optimized",
        "harm_direct",
    ),
    Condition(
        "harm_choice_q397_specialist_empty",
        "harm_choice",
        "q397_optimized",
        "harm_empty",
    ),
    Condition(
        "harm_choice_base_specialist_direct",
        "harm_choice",
        "base_qwen",
        "harm_direct",
    ),
    Condition(
        "harm_choice_base_specialist_empty",
        "harm_choice",
        "base_qwen",
        "harm_empty",
    ),
    Condition(
        "intent_validation_q397_output_only",
        "intent_validation",
        "q397_optimized",
        "general",
    ),
    Condition(
        "intent_validation_q397_reasoning_visible",
        "intent_validation",
        "q397_optimized",
        "general_reasoning",
    ),
    Condition(
        "intent_validation_q397_intent_direct",
        "intent_validation",
        "q397_optimized",
        "intent_direct",
    ),
    Condition(
        "intent_validation_q397_intent_empty",
        "intent_validation",
        "q397_optimized",
        "intent_empty",
    ),
    Condition(
        "intent_validation_current_rank1",
        "intent_validation",
        "current_intent_rank1",
        "intent_empty",
    ),
    Condition(
        "intent_test_q397_output_only",
        "intent_test",
        "q397_optimized",
        "general",
    ),
    Condition(
        "intent_test_q397_reasoning_visible",
        "intent_test",
        "q397_optimized",
        "general_reasoning",
    ),
    Condition(
        "intent_test_q397_intent_direct",
        "intent_test",
        "q397_optimized",
        "intent_direct",
    ),
    Condition(
        "intent_test_q397_intent_empty",
        "intent_test",
        "q397_optimized",
        "intent_empty",
    ),
    Condition(
        "intent_test_current_rank1",
        "intent_test",
        "current_intent_rank1",
        "intent_empty",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument(
        "--external-artifact",
        type=Path,
        default=DEFAULT_EXTERNAL_ARTIFACT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--remote-batches-per-session", type=int, default=0)
    parser.add_argument(
        "--adapter",
        action="append",
        choices=sorted(ADAPTERS),
        help="Repeat to score a subset; defaults to all adapters.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=[condition.name for condition in CONDITIONS],
        help="Repeat to score a subset; defaults to all conditions.",
    )
    return parser.parse_args()


def load_credentials() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    os.environ.setdefault("NDIF_HOST", DEFAULT_NDIF_HOST)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is missing")


@lru_cache(maxsize=1)
def binary_prompt() -> str:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "configs"),
    ):
        config = compose(
            config_name="pid_qwen397_tvg_binary_soft_distillation_v1"
        )
    prompt = str(config.student.prompt)
    if not isinstance(prompt, str) or "Prediction:<0 or 1>" not in prompt:
        raise ValueError("optimized Qwen-397B binary prompt contract changed")
    return prompt


def final_reasoning(messages: Any) -> str:
    normalized = list(messages or [])
    for message in reversed(normalized):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("reasoning") or "")
    return ""


def add_reasoning(member_prompt: str, messages: Any) -> str:
    reasoning = final_reasoning(messages)
    if not reasoning:
        return member_prompt
    return (
        f"{member_prompt}\n\n<assistant_reasoning>\n"
        f"{reasoning[-REASONING_MAX_CHARS:]}\n"
        "</assistant_reasoning>"
    )


def chat_boundary(
    tokenizer: Any,
    member_prompt: str,
    *,
    empty_summary: bool,
) -> str:
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": member_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return chat + (EMPTY_REASONING_PREFIX if empty_summary else "Prediction:")


def build_general_member(messages: Any) -> str:
    rendered, _ = build_structural_member_prompt(
        messages,
        PHOENIX_CAP_CONDITIONS[0],
    )
    _, separator, evidence = rendered.partition("<context>")
    if not separator:
        raise ValueError("Phoenix structural renderer omitted <context>")
    return f"{binary_prompt()}\n\n<context>{evidence}"


def build_harm_member(messages: Any) -> str:
    return build_student_prompt(
        messages,
        HARM_PRESSURE_CHOICE_PROMPT,
        4_500,
        "head_tail",
        include_reasoning=False,
    )


def prompt_builder(kind: str) -> Callable[[Any, Any], str]:
    builders: dict[str, Callable[[Any, Any], str]] = {
        "general": lambda messages, tokenizer: chat_boundary(
            tokenizer,
            build_general_member(messages),
            empty_summary=False,
        ),
        "general_reasoning": lambda messages, tokenizer: chat_boundary(
            tokenizer,
            add_reasoning(build_general_member(messages), messages),
            empty_summary=False,
        ),
        "action_direct": lambda messages, tokenizer: chat_boundary(
            tokenizer,
            build_action_member_prompt(messages),
            empty_summary=False,
        ),
        "action_empty": lambda messages, tokenizer: chat_boundary(
            tokenizer,
            build_action_member_prompt(messages),
            empty_summary=True,
        ),
        "hpkr_direct": lambda messages, tokenizer: chat_boundary(
            tokenizer,
            build_epistemic_member_prompt(messages),
            empty_summary=False,
        ),
        "hpkr_empty": lambda messages, tokenizer: chat_boundary(
            tokenizer,
            build_epistemic_member_prompt(messages),
            empty_summary=True,
        ),
        "harm_direct": lambda messages, tokenizer: chat_boundary(
            tokenizer,
            build_harm_member(messages),
            empty_summary=False,
        ),
        "harm_empty": lambda messages, tokenizer: chat_boundary(
            tokenizer,
            build_harm_member(messages),
            empty_summary=True,
        ),
        "intent_direct": lambda messages, tokenizer: chat_boundary(
            tokenizer,
            build_intent_member_prompt(messages),
            empty_summary=False,
        ),
        "intent_empty": lambda messages, tokenizer: chat_boundary(
            tokenizer,
            build_intent_member_prompt(messages),
            empty_summary=True,
        ),
    }
    try:
        return builders[kind]
    except KeyError as error:
        raise ValueError(f"unknown prompt kind: {kind}") from error


def external_half(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    development = {
        (str(row["dataset"]), str(row["index"]))
        for row in split_records(rows, "development", seed=seed)
    }
    output = []
    for row in rows:
        copied = dict(row)
        key = (str(row["dataset"]), str(row["index"]))
        copied["half"] = (
            "development" if key in development else "confirmation"
        )
        output.append(copied)
    return output


def load_external_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    declarations = {
        "action": ("insider-trading", 20260726),
        "hpkr": ("harm-pressure-knowledge-report", 20260725),
        "harm_choice": ("harm-pressure-choice", 20260725),
    }
    output = {}
    for group, (category, seed) in declarations.items():
        selected = [row for row in rows if row["category"] == category]
        if len(selected) != 200:
            raise RuntimeError(
                f"{category}: expected 200 frozen rows, found {len(selected)}"
            )
        output[group] = external_half(selected, seed=seed)
    return output


def load_competition_records(
    split: str,
    splits_dir: Path,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    manifest = yaml.safe_load((splits_dir / f"dry.{split}.yaml").read_text())
    rows: list[dict[str, Any]] = []
    for declaration in manifest["datasets"]:
        dataset_name = str(declaration["name"])
        if "varied-deception" not in dataset_name:
            continue
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
        found = set()
        for row in dataset:
            index = row["index"]
            if index not in label_by_index:
                continue
            messages = row["messages"]
            if not has_final_assistant_reasoning(messages):
                continue
            found.add(index)
            rows.append({
                "dataset": dataset_name,
                "index": index,
                "label": int(label_by_index[index]),
                "messages": messages,
                "split": split,
            })
        if found != set(label_by_index):
            raise RuntimeError(
                f"{dataset_name}: reasoning-bearing coverage "
                f"{len(found)}/{len(label_by_index)}"
            )
    if len(rows) != 360 or len({row["dataset"] for row in rows}) != 9:
        raise RuntimeError(
            f"{split}: expected 360 rows across 9 units, found "
            f"{len(rows)} across {len({row['dataset'] for row in rows})}"
        )
    return rows


def load_record_groups(
    external_artifact: Path,
    splits_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    groups = load_external_records(external_artifact)
    groups["intent_validation"] = load_competition_records(
        "validation",
        splits_dir,
    )
    groups["intent_test"] = load_competition_records("test", splits_dir)
    return groups


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


def position_batches(lengths: list[int]) -> list[list[int]]:
    order = np.argsort(lengths)
    batches = []
    cursor = 0
    while cursor < len(order):
        cap = 48
        candidate = order[cursor : min(cursor + cap, len(order))]
        if max(lengths[position] for position in candidate) > 600:
            cap = 32
            candidate = order[cursor : min(cursor + cap, len(order))]
        if max(lengths[position] for position in candidate) > 900:
            cap = 16
            candidate = order[cursor : min(cursor + cap, len(order))]
        batches.append(candidate.tolist())
        cursor += len(candidate)
    return batches


def encode_prompts(
    tokenizer: Any,
    prompts: list[str],
) -> tuple[list[tuple[list[int], Any]], dict[str, Any]]:
    lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    ]
    encoded = []
    for positions in position_batches(lengths):
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
        "minimum_tokens": min(lengths),
        "median_tokens": float(np.median(lengths)),
        "p95_tokens": float(np.quantile(lengths, 0.95)),
        "maximum_tokens": max(lengths),
        "truncated_rows": sum(
            length > MAX_PROMPT_TOKENS for length in lengths
        ),
        "batch_shapes": [
            [len(positions), int(tokens["input_ids"].shape[1])]
            for positions, tokens in encoded
        ],
    }


def score_adapter(
    spec: AdapterSpec,
    conditions: list[Condition],
    prompts: dict[str, list[str]],
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

    flat_batches = []
    audits = {}
    scores = {}
    for condition in conditions:
        batches, audit = encode_prompts(
            tokenizer,
            prompts[condition.name],
        )
        audits[condition.name] = audit
        scores[condition.name] = np.full(
            len(prompts[condition.name]),
            np.nan,
            dtype=np.float64,
        )
        flat_batches.extend(
            (condition.name, positions, tokens)
            for positions, tokens in batches
        )

    group_size = remote_batches_per_session or len(flat_batches)
    started = time.perf_counter()
    for group_start in range(0, len(flat_batches), group_size):
        group = flat_batches[group_start : group_start + group_size]
        print(
            f"adapter={spec.name} batches={group_start + 1}-"
            f"{group_start + len(group)}/{len(flat_batches)}",
            flush=True,
        )
        pieces = []
        with model.session(remote=True):
            for _, _, tokens in group:
                with model.trace({
                    "input_ids": tokens["input_ids"],
                    "attention_mask": tokens["attention_mask"],
                    "logits_to_keep": 1,
                }):
                    logits = model.output.logits[:, -1, label_ids].float()
                    pieces.append(
                        torch.softmax(logits, dim=-1)[:, 1]
                        .detach()
                        .cpu()
                    )
            values_proxy = torch.cat(pieces, dim=0).save()
        values = np.asarray(values_proxy.float().tolist(), dtype=np.float64)
        cursor = 0
        for condition_name, positions, _ in group:
            count = len(positions)
            scores[condition_name][positions] = values[cursor : cursor + count]
            cursor += count

    for name, values in scores.items():
        if np.isnan(values).any():
            raise RuntimeError(
                f"{name}: {int(np.isnan(values).sum())} missing NDIF scores"
            )
    return scores, {
        "elapsed_seconds": time.perf_counter() - started,
        "label_token_ids": label_ids,
        "prompt_audits": audits,
    }


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = scores >= threshold
    positive = labels == 1
    negative = labels == 0
    recall = float(predictions[positive].mean()) if positive.any() else None
    fpr = float(predictions[negative].mean()) if negative.any() else None
    auroc = (
        float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else None
    )
    balanced_accuracy = (
        (recall + 1.0 - fpr) / 2.0
        if recall is not None and fpr is not None
        else None
    )
    counts = Counter(float(score) for score in scores)
    return {
        "rows": int(len(labels)),
        "auroc": auroc,
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "fpr": fpr,
        "unique_scores": len(counts),
        "duplicate_score_excess_rows": int(len(scores) - len(counts)),
        "rows_participating_in_ties": int(
            sum(count for count in counts.values() if count > 1)
        ),
    }


def grouped_metrics(
    frame: pd.DataFrame,
    field: str,
) -> dict[str, dict[str, float | int | None]]:
    return {
        str(name): binary_metrics(
            group["label"].to_numpy(),
            group["score"].to_numpy(),
        )
        for name, group in frame.groupby(field, sort=True)
    }


def macro_auroc(groups: dict[str, dict[str, Any]]) -> float | None:
    values = [
        float(metrics["auroc"])
        for metrics in groups.values()
        if metrics["auroc"] is not None
    ]
    return float(np.mean(values)) if values else None


def metric_bundle(
    records: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "messages"}
         for row in records]
    )
    frame["score"] = scores
    per_dataset = grouped_metrics(frame, "dataset")
    per_source = (
        grouped_metrics(frame, "source_model")
        if "source_model" in frame
        else {}
    )
    result: dict[str, Any] = {
        "all": {
            **binary_metrics(
                frame["label"].to_numpy(),
                frame["score"].to_numpy(),
            ),
            "macro_dataset_auroc": macro_auroc(per_dataset),
            "macro_source_auroc": macro_auroc(per_source),
        },
        "per_dataset": per_dataset,
        "per_source_model": per_source,
    }
    if "half" in frame:
        result["halves"] = {
            str(name): binary_metrics(
                group["label"].to_numpy(),
                group["score"].to_numpy(),
            )
            for name, group in frame.groupby("half", sort=True)
        }
    return result


def log_odds_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.clip(np.asarray(left, dtype=float), EPSILON, 1.0 - EPSILON)
    right = np.clip(np.asarray(right, dtype=float), EPSILON, 1.0 - EPSILON)
    margin = 0.5 * (
        np.log(left / (1.0 - left))
        + np.log(right / (1.0 - right))
    )
    return 1.0 / (1.0 + np.exp(-margin))


def write_predictions(
    output_dir: Path,
    condition_name: str,
    records: list[dict[str, Any]],
    scores: np.ndarray,
) -> None:
    frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "messages"}
         for row in records]
    )
    frame["score"] = scores
    frame.to_csv(output_dir / f"{condition_name}.csv", index=False)


def prompt_audit(
    record_groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    audit = {}
    for group in ("intent_validation", "intent_test"):
        lengths = [
            len(final_reasoning(row["messages"]))
            for row in record_groups[group]
        ]
        audit[group] = {
            "rows": len(lengths),
            "nonempty_reasoning_rows": sum(length > 0 for length in lengths),
            "minimum_reasoning_chars": min(lengths),
            "median_reasoning_chars": float(np.median(lengths)),
            "p95_reasoning_chars": float(np.quantile(lengths, 0.95)),
            "maximum_reasoning_chars": max(lengths),
            "reasoning_char_truncated_rows": sum(
                length > REASONING_MAX_CHARS for length in lengths
            ),
            "retained_reasoning_chars": REASONING_MAX_CHARS,
            "reasoning_truncation": "tail",
        }
    return audit


def main() -> None:
    args = parse_args()
    load_credentials()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    record_groups = load_record_groups(
        args.external_artifact.resolve(),
        args.splits_dir.resolve(),
    )

    selected_names = set(args.condition or [item.name for item in CONDITIONS])
    selected_adapters = set(args.adapter or ADAPTERS)
    selected_conditions = [
        item
        for item in CONDITIONS
        if item.name in selected_names and item.adapter in selected_adapters
    ]
    if not selected_conditions:
        raise ValueError("no conditions selected")

    from transformers import AutoTokenizer

    rendering_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = {
        condition.name: [
            prompt_builder(condition.prompt_kind)(
                row["messages"],
                rendering_tokenizer,
            )
            for row in record_groups[condition.record_group]
        ]
        for condition in selected_conditions
    }
    report: dict[str, Any] = {
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "backend": "competition_ndif",
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "reasoning_max_chars": REASONING_MAX_CHARS,
        "conditions": {},
        "adapters": {},
        "intent_reasoning_audit": prompt_audit(record_groups),
        "prompt_sha256": {
            name: hashlib.sha256(
                "\0".join(values).encode("utf-8")
            ).hexdigest()
            for name, values in prompts.items()
        },
    }
    all_scores: dict[str, np.ndarray] = {}
    for adapter_name in ADAPTERS:
        adapter_conditions = [
            item
            for item in selected_conditions
            if item.adapter == adapter_name
        ]
        if not adapter_conditions:
            continue
        spec = ADAPTERS[adapter_name]
        print(
            f"starting adapter={adapter_name} repo={spec.repo_id} "
            f"conditions={len(adapter_conditions)}",
            flush=True,
        )
        adapter_scores, timing = score_adapter(
            spec,
            adapter_conditions,
            prompts,
            remote_batches_per_session=args.remote_batches_per_session,
        )
        report["adapters"][adapter_name] = {
            "spec": asdict(spec),
            "timing": timing,
        }
        for condition in adapter_conditions:
            values = adapter_scores[condition.name]
            all_scores[condition.name] = values
            metrics = metric_bundle(
                record_groups[condition.record_group],
                values,
            )
            report["conditions"][condition.name] = {
                "spec": asdict(condition),
                "metrics": metrics,
            }
            write_predictions(
                output_dir,
                condition.name,
                record_groups[condition.record_group],
                values,
            )
            print(
                f"finished condition={condition.name} "
                f"auroc={metrics['all']['auroc']:.6f} "
                f"macro_dataset={metrics['all']['macro_dataset_auroc']:.6f} "
                f"unique={metrics['all']['unique_scores']}",
                flush=True,
            )
        (output_dir / "report.partial.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

    for split in ("validation", "test"):
        group = f"intent_{split}"
        main_name = f"{group}_q397_output_only"
        reasoning_name = f"{group}_q397_reasoning_visible"
        intent_name = f"{group}_current_rank1"
        if not {main_name, reasoning_name, intent_name} <= all_scores.keys():
            continue
        derived = {
            f"{group}_current_route_blend": log_odds_mean(
                all_scores[main_name],
                all_scores[intent_name],
            ),
            f"{group}_reasoning_plus_rank1_blend": log_odds_mean(
                all_scores[reasoning_name],
                all_scores[intent_name],
            ),
        }
        for name, values in derived.items():
            all_scores[name] = values
            metrics = metric_bundle(record_groups[group], values)
            report["conditions"][name] = {
                "spec": {
                    "name": name,
                    "record_group": group,
                    "adapter": "derived",
                    "prompt_kind": "mean_log_odds",
                },
                "metrics": metrics,
            }
            write_predictions(output_dir, name, record_groups[group], values)

    comparisons = {
        "action_specialist_vs_q397_general": (
            "action_q397_general_direct",
            "action_current_specialist",
        ),
        "action_specialist_vs_q397_action_empty": (
            "action_q397_action_empty",
            "action_current_specialist",
        ),
        "hpkr_specialist_vs_q397_general": (
            "hpkr_q397_general_direct",
            "hpkr_current_base_specialist",
        ),
        "hpkr_specialist_vs_q397_epistemic_empty": (
            "hpkr_q397_epistemic_empty",
            "hpkr_current_base_specialist",
        ),
        "harm_q397_empty_prompt_vs_general": (
            "harm_choice_q397_general_direct",
            "harm_choice_q397_specialist_empty",
        ),
        "intent_validation_reasoning_vs_output": (
            "intent_validation_q397_output_only",
            "intent_validation_q397_reasoning_visible",
        ),
        "intent_test_reasoning_vs_output": (
            "intent_test_q397_output_only",
            "intent_test_q397_reasoning_visible",
        ),
        "intent_validation_route_vs_reasoning": (
            "intent_validation_q397_reasoning_visible",
            "intent_validation_current_route_blend",
        ),
        "intent_test_route_vs_reasoning": (
            "intent_test_q397_reasoning_visible",
            "intent_test_current_route_blend",
        ),
        "intent_validation_reasoning_rank1_vs_reasoning": (
            "intent_validation_q397_reasoning_visible",
            "intent_validation_reasoning_plus_rank1_blend",
        ),
        "intent_test_reasoning_rank1_vs_reasoning": (
            "intent_test_q397_reasoning_visible",
            "intent_test_reasoning_plus_rank1_blend",
        ),
    }
    report["comparisons"] = {}
    for name, (left, right) in comparisons.items():
        if left not in report["conditions"] or right not in report["conditions"]:
            continue
        left_metrics = report["conditions"][left]["metrics"]["all"]
        right_metrics = report["conditions"][right]["metrics"]["all"]
        metric = (
            "macro_dataset_auroc"
            if left.startswith("intent_")
            else "auroc"
        )
        report["comparisons"][name] = {
            "left": left,
            "right": right,
            "metric": metric,
            "left_value": left_metrics[metric],
            "right_value": right_metrics[metric],
            "right_minus_left": (
                float(right_metrics[metric]) - float(left_metrics[metric])
            ),
        }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    partial = output_dir / "report.partial.json"
    if partial.exists():
        partial.unlink()
    print(json.dumps({
        "report": report_path.as_posix(),
        "comparisons": report["comparisons"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
