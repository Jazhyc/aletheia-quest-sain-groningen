#!/usr/bin/env python3
"""Evaluate a rank-1 Qwen fact retriever against frozen GPT-OSS labels."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.privileged_information_distillation.evaluate_student_sft import (
    binary_token_ids,
    score_binary_prefixes,
)
from experiments.wikidata_rag.build_qwen_retriever_distillation import (
    DECISIVE,
    group_bucket,
    load_jsonl,
    retrieval_prompt,
)
from experiments.wikidata_rag.qwen_database_planner import evaluate_plans
from experiments.wikidata_rag.train_gptoss_fact_ranker import (
    choose_threshold,
    row_operating_point,
)


def valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep rows with complete teacher labels and at least one candidate."""
    return [
        row for row in rows
        if not row.get("parse_error") and row.get("candidates") and row.get("labels")
    ]


def candidate_labels(row: dict[str, Any]) -> list[int]:
    """Return teacher decisiveness labels in candidate order."""
    by_id = {
        str(item["id"]): str(item["label"]) for item in row.get("labels", [])
    }
    return [
        int(by_id.get(str(candidate["id"])) in DECISIVE)
        for candidate in row["candidates"]
    ]


def flatten_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> tuple[list[str], np.ndarray, list[tuple[int, int]]]:
    """Render candidate prompts and retain row slices."""
    prompts: list[str] = []
    labels: list[int] = []
    slices: list[tuple[int, int]] = []
    for row in rows:
        start = len(prompts)
        for candidate, label in zip(
            row["candidates"], candidate_labels(row), strict=True
        ):
            raw = retrieval_prompt(row, candidate)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(rendered + "Prediction:")
            labels.append(label)
        slices.append((start, len(prompts)))
    return prompts, np.asarray(labels, dtype=np.int8), slices


def score_in_batches(
    llm: Any,
    prompts: list[str],
    sampling: Any,
    request: Any,
    token_ids: list[int],
    batch_size: int,
) -> tuple[np.ndarray, int, float]:
    """Score binary margins without retaining every vLLM output at once."""
    scores: list[float] = []
    missing = 0
    elapsed = 0.0
    for start in range(0, len(prompts), batch_size):
        values, batch_missing, seconds = score_binary_prefixes(
            llm,
            prompts[start:start + batch_size],
            sampling,
            request,
            token_ids,
        )
        scores.extend(values)
        missing += batch_missing
        elapsed += seconds
    return np.asarray(scores, dtype=np.float64), missing, elapsed


def safe_metric(metric: Any, labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Return a binary ranking metric when both labels are present."""
    if not len(labels) or len(np.unique(labels)) < 2:
        return None
    return float(metric(labels, scores))


def subset_report(
    scores: np.ndarray,
    labels: np.ndarray,
    row_slices: list[tuple[int, int]],
    row_ids: list[int],
    *,
    absolute_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    """Report candidate ranking and frozen row-level operating points."""
    candidate_ids = np.concatenate([
        np.arange(*row_slices[row_id], dtype=int) for row_id in row_ids
    ]) if row_ids else np.asarray([], dtype=int)
    top1_hits = 0
    rows_with_decisive = 0
    for row_id in row_ids:
        start, end = row_slices[row_id]
        row_labels = labels[start:end]
        rows_with_decisive += int(bool(row_labels.any()))
        if len(row_labels):
            top1_hits += int(row_labels[int(np.argmax(scores[start:end]))])
    return {
        "rows": len(row_ids),
        "candidates": len(candidate_ids),
        "decisive_candidates": int(labels[candidate_ids].sum()) if len(candidate_ids) else 0,
        "rows_with_decisive": rows_with_decisive,
        "candidate_auroc": safe_metric(
            roc_auc_score, labels[candidate_ids], scores[candidate_ids]
        ),
        "candidate_average_precision": safe_metric(
            average_precision_score, labels[candidate_ids], scores[candidate_ids]
        ),
        "top1_decisive_rows": top1_hits,
        "top1_recall_of_decisive_rows": top1_hits / max(1, rows_with_decisive),
        "top1_precision_over_all_rows": top1_hits / max(1, len(row_ids)),
        "absolute": row_operating_point(
            scores,
            labels,
            row_slices,
            row_ids,
            absolute_threshold,
            "absolute",
        ),
        "margin": row_operating_point(
            scores,
            labels,
            row_slices,
            row_ids,
            margin_threshold,
            "margin",
        ),
    }


def split_row_ids(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> dict[str, list[int]]:
    """Build frozen exact-question grouped train and validation subsets."""
    train_count = len(train_rows)
    training_groups = {str(row["question_group"]) for row in train_rows}
    frozen = list(range(train_count, train_count + len(validation_rows)))
    return {
        "fit": [
            index for index, row in enumerate(train_rows)
            if group_bucket(str(row["question_group"])) in {2, 3, 4}
        ],
        "calibration": [
            index for index, row in enumerate(train_rows)
            if group_bucket(str(row["question_group"])) == 1
        ],
        "internal_test": [
            index for index, row in enumerate(train_rows)
            if group_bucket(str(row["question_group"])) == 0
        ],
        "validation": frozen,
        "validation_novel": [
            index for index in frozen
            if str((train_rows + validation_rows)[index]["question_group"])
            not in training_groups
        ],
        "validation_seen": [
            index for index in frozen
            if str((train_rows + validation_rows)[index]["question_group"])
            in training_groups
        ],
    }


def calibrated_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    row_slices: list[tuple[int, int]],
    calibration_ids: list[int],
    minimum_precision: float,
) -> dict[str, dict[str, Any]]:
    """Select absolute and runner-up-margin thresholds on training calibration."""
    output = {}
    for mode in ("absolute", "margin"):
        threshold, report = choose_threshold(
            scores,
            labels,
            row_slices,
            calibration_ids,
            minimum_precision,
            mode,
        )
        output[mode] = {"threshold": threshold, "calibration": report}
    return output


def filtered_plans(
    planner_rows: list[dict[str, Any]],
    score_by_key_id: dict[tuple[str, int, str], float],
    threshold: float,
) -> list[dict[str, Any]]:
    """Filter grounded planner selections with an independently trained score."""
    output = []
    for row in planner_rows:
        selected = [
            item for item in row.get("selected", [])
            if score_by_key_id.get(
                (str(row["dataset"]), int(row["index"]), str(item["id"])),
                -float("inf"),
            ) >= threshold
        ]
        filtered = copy.deepcopy(row)
        filtered["selected"] = selected
        output.append(filtered)
    return output


def independent_plans(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    row_slices: list[tuple[int, int]],
    row_offset: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Select the top candidate above a frozen absolute threshold."""
    output = []
    for local_id, row in enumerate(rows):
        start, end = row_slices[row_offset + local_id]
        selected = []
        if start < end:
            local = int(np.argmax(scores[start:end]))
            if float(scores[start + local]) >= threshold:
                selected = [{"id": str(row["candidates"][local]["id"])}]
        output.append({**row, "selected": selected, "parse_error": False})
    return output


def grounding_inputs(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    row_slices: list[tuple[int, int]],
    row_offset: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Build label-blind one-candidate rows for a later quote/relation pass."""
    output = []
    for local_id, row in enumerate(rows):
        start, end = row_slices[row_offset + local_id]
        if start == end:
            continue
        local = int(np.argmax(scores[start:end]))
        score = float(scores[start + local])
        if score < threshold:
            continue
        candidate = copy.deepcopy(row["candidates"][local])
        stripped = {
            key: copy.deepcopy(value)
            for key, value in row.items()
            if key not in {"labels", "teacher_model", "rendered_prompt", "raw_completion"}
        }
        stripped["candidates"] = [candidate]
        stripped["retriever_score"] = score
        output.append(stripped)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--validation-planner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--adapter-only",
        action="store_true",
        help="Skip the already-frozen base-Qwen control for follow-up adapters.",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    train_rows = valid_rows(load_jsonl(args.train_labels))
    validation_rows = valid_rows(load_jsonl(args.validation_labels))
    rows = train_rows + validation_rows
    splits = split_row_ids(train_rows, validation_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    prompts, labels, row_slices = flatten_rows(rows, tokenizer)
    token_ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(token_ids),
        logprob_token_ids=token_ids,
        allowed_token_ids=token_ids,
    )
    llm = LLM(
        model="Qwen/Qwen3.5-9B",
        tokenizer=args.adapter_dir.as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
    )
    request = LoRARequest(
        args.adapter_dir.parent.name,
        1,
        args.adapter_dir.as_posix(),
    )
    started = time.time()
    base_scores = None
    base_missing = 0
    base_seconds = 0.0
    if not args.adapter_only:
        base_scores, base_missing, base_seconds = score_in_batches(
            llm, prompts, sampling, None, token_ids, args.batch_size
        )
    adapter_scores, adapter_missing, adapter_seconds = score_in_batches(
        llm, prompts, sampling, request, token_ids, args.batch_size
    )

    model_scores = {"adapter": adapter_scores}
    if base_scores is not None:
        model_scores = {"base": base_scores, **model_scores}
    thresholds = {
        name: calibrated_thresholds(
            scores,
            labels,
            row_slices,
            splits["calibration"],
            args.minimum_precision,
        )
        for name, scores in model_scores.items()
    }
    reports = {
        name: {
            split: subset_report(
                scores,
                labels,
                row_slices,
                row_ids,
                absolute_threshold=thresholds[name]["absolute"]["threshold"],
                margin_threshold=thresholds[name]["margin"]["threshold"],
            )
            for split, row_ids in splits.items()
        }
        for name, scores in model_scores.items()
    }

    validation_offset = len(train_rows)
    planner_rows = load_jsonl(args.validation_planner)
    validation_score_maps = {}
    for name, scores in model_scores.items():
        mapping = {}
        for local_id, row in enumerate(validation_rows):
            start, end = row_slices[validation_offset + local_id]
            for candidate, score in zip(
                row["candidates"], scores[start:end], strict=True
            ):
                mapping[(
                    str(row["dataset"]),
                    int(row["index"]),
                    str(candidate["id"]),
                )] = float(score)
        validation_score_maps[name] = mapping

    planner_reports = {
        "unfiltered": evaluate_plans(planner_rows, validation_rows),
    }
    for name in model_scores:
        threshold = thresholds[name]["absolute"]["threshold"]
        planner_reports[f"filtered_{name}"] = evaluate_plans(
            filtered_plans(
                planner_rows,
                validation_score_maps[name],
                threshold,
            ),
            validation_rows,
        )
        planner_reports[f"independent_{name}"] = evaluate_plans(
            independent_plans(
                validation_rows,
                model_scores[name],
                row_slices,
                validation_offset,
                threshold,
            ),
            validation_rows,
        )

    result = {
        "adapter_dir": args.adapter_dir.resolve().as_posix(),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "candidate_prompts": len(prompts),
        "minimum_calibration_precision": args.minimum_precision,
        "thresholds": thresholds,
        "models": reports,
        "planner": planner_reports,
        "missing_logits": {"base": base_missing, "adapter": adapter_missing},
        "score_seconds": {"base": base_seconds, "adapter": adapter_seconds},
        "wall_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    scored_validation = []
    for local_id, row in enumerate(validation_rows):
        start, end = row_slices[validation_offset + local_id]
        for offset, (candidate, label, adapter_score) in enumerate(zip(
            row["candidates"], labels[start:end], adapter_scores[start:end], strict=True
        )):
            record = {
                "dataset": row["dataset"],
                "index": row["index"],
                "question_group": row["question_group"],
                "candidate_id": candidate["id"],
                "subject": candidate["subject"],
                "fact": candidate["fact"],
                "teacher_decisive": int(label),
                "adapter_score": float(adapter_score),
            }
            if base_scores is not None:
                record["base_score"] = float(base_scores[start + offset])
            scored_validation.append(record)
    (args.output_dir / "validation_candidate_scores.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in scored_validation) + "\n"
    )
    selected_for_grounding = grounding_inputs(
        validation_rows,
        adapter_scores,
        row_slices,
        validation_offset,
        thresholds["adapter"]["absolute"]["threshold"],
    )
    (args.output_dir / "validation_grounding_input.jsonl").write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False) for row in selected_for_grounding
        ) + ("\n" if selected_for_grounding else "")
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
