#!/usr/bin/env python3
"""Materialize Phoenix 8.1 prompts and labels from MIT annotations.

The generated local file contains text fetched from the cited source datasets
and therefore remains subject to those datasets' licenses. It is deliberately
not part of the MIT annotation repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


TRUNCATION_MARKER = "\n\n[... middle truncated ...]\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/train-00000-of-00001.parquet"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path(__file__).with_name("prompt_templates.json"),
    )
    parser.add_argument(
        "--include-messages",
        action="store_true",
        help="Also copy normalized source messages into the local output.",
    )
    return parser.parse_args()


def safe_text(value: Any) -> str:
    return str(value if value is not None else "").encode(
        "utf-8", errors="replace"
    ).decode("utf-8", errors="replace")


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        messages = [{"role": "unknown", "content": messages}]
    normalized = []
    for message in messages:
        if hasattr(message, "as_py"):
            message = message.as_py()
        if not isinstance(message, dict):
            message = {"role": "unknown", "content": message}
        normalized.append(
            {
                "role": safe_text(message.get("role", "unknown")),
                "content": safe_text(message.get("content", "")),
            }
        )
    return normalized


def merge_messages(messages: Any) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for message in normalize_messages(messages):
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"] += "\n" + message["content"]
        else:
            merged.append(dict(message))
    return merged


def truncate_text(text: str, max_chars: int, mode: str) -> str:
    if len(text) <= max_chars:
        return text
    if mode == "tail":
        return text[-max_chars:]
    if mode == "head":
        return text[:max_chars]
    if mode == "head_tail":
        available = max(0, max_chars - len(TRUNCATION_MARKER))
        head = int(available * 0.5)
        return text[:head] + TRUNCATION_MARKER + text[-(available - head) :]
    raise ValueError(f"unknown truncation mode: {mode}")


def render_prompt(
    messages: Any,
    template: str,
    max_chars: int,
    truncation: str,
) -> str:
    merged = merge_messages(messages)
    final_assistant = next(
        (
            index
            for index in range(len(merged) - 1, -1, -1)
            if merged[index]["role"] == "assistant"
        ),
        None,
    )
    output = merged[final_assistant]["content"] if final_assistant is not None else ""
    context = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in merged
    )
    context = truncate_text(context, int(max_chars), str(truncation))
    return (
        f"{template}\n\n<context>\n{context}\n</context>"
        f"\n\n<output>\n{output}\n</output>"
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parquet_filename(config: str, split: str) -> str:
    return (
        f"data/{split}-00000-of-00001.parquet"
        if config in {"", "default", "None", "nan"}
        else f"{config}/{split}-00000-of-00001.parquet"
    )


def download_frame(repo_id: str, config: str, revision: str, split: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=parquet_filename(config, split),
        repo_type="dataset",
        revision=revision,
    )
    return pd.read_parquet(path)


def load_source_rows(annotations: pd.DataFrame) -> dict[tuple[str, str, str, str], dict]:

    loaded: dict[tuple[str, str, str, str], dict] = {}
    groups = annotations.groupby(
        ["source_dataset", "source_config", "source_revision", "source_split"],
        sort=True,
        dropna=False,
    )
    for (repo_id, config, revision, split), _ in groups:
        frame = download_frame(
            str(repo_id), str(config), str(revision), str(split)
        )
        for row in frame.to_dict(orient="records"):
            loaded[(str(repo_id), str(config), str(revision), str(row["index"]))] = row
    return loaded


def load_label_rows(annotations: pd.DataFrame) -> dict[tuple[str, str, str, str], int]:
    loaded: dict[tuple[str, str, str, str], int] = {}
    groups = annotations.groupby(
        [
            "source_label_dataset",
            "source_label_config",
            "source_label_revision",
            "source_split",
        ],
        sort=True,
        dropna=False,
    )
    for (repo_id, config, revision, split), _ in groups:
        frame = download_frame(
            str(repo_id), str(config), str(revision), str(split)
        )
        for row in frame.to_dict(orient="records"):
            label = row.get("deceptive", row.get("label"))
            if label is None:
                raise ValueError(f"label source {repo_id}/{config} has no label column")
            loaded[(str(repo_id), str(config), str(revision), str(row["index"]))] = int(label)
    return loaded


def materialize(
    annotations: pd.DataFrame,
    templates: dict[str, str],
    *,
    include_messages: bool = False,
) -> pd.DataFrame:
    source_rows = load_source_rows(annotations)
    label_rows = load_label_rows(annotations)
    materialized = []
    for annotation in annotations.to_dict(orient="records"):
        source_key = (
            str(annotation["source_dataset"]),
            str(annotation["source_config"]),
            str(annotation["source_revision"]),
            str(annotation["source_index"]),
        )
        label_key = (
            str(annotation["source_label_dataset"]),
            str(annotation["source_label_config"]),
            str(annotation["source_label_revision"]),
            str(annotation["source_index"]),
        )
        if source_key not in source_rows:
            raise KeyError(f"source row not found: {source_key}")
        if label_key not in label_rows:
            raise KeyError(f"label row not found: {label_key}")
        messages = normalize_messages(source_rows[source_key]["messages"])
        student_prompt = render_prompt(
            messages,
            templates[str(annotation["student_prompt_kind"])],
            int(annotation["student_context_chars"]),
            str(annotation["student_context_truncation"]),
        )
        teacher_prompt = render_prompt(
            messages,
            templates[str(annotation["teacher_prompt_kind"])],
            int(annotation["teacher_context_chars"]),
            str(annotation["teacher_context_truncation"]),
        )
        if sha256_text(student_prompt) != str(annotation["student_prompt_sha256"]):
            raise ValueError(f"student prompt hash mismatch for {annotation['row_id']}")
        if sha256_text(teacher_prompt) != str(annotation["teacher_prompt_sha256"]):
            raise ValueError(f"teacher prompt hash mismatch for {annotation['row_id']}")
        record = {
            **annotation,
            "label": label_rows[label_key],
            "student_prompt": student_prompt,
            "teacher_prompt": teacher_prompt,
        }
        if include_messages:
            record["messages"] = messages
        materialized.append(record)
    return pd.DataFrame(materialized)


def main() -> None:
    args = parse_args()
    annotations = pd.read_parquet(args.annotations)
    templates = json.loads(args.templates.read_text())
    output = materialize(
        annotations,
        templates,
        include_messages=args.include_messages,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    print(f"wrote {len(output)} verified rows to {args.output}")


if __name__ == "__main__":
    main()
