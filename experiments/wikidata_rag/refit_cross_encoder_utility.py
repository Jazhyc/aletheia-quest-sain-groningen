#!/usr/bin/env python3
"""Refit a selected MiniLM utility policy on more groups and average seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.wikidata_rag.train_cross_encoder_utility import (
    RowDataset,
    candidate_indices,
    candidate_targets,
    collate_rows,
    evaluate_checkpoint,
    freeze_bottom_layers,
    group_weights,
    linear_schedule,
    load_jsonl,
    load_model,
    predict_rows,
    row_slices,
    save_best,
    split_row_ids,
    target_training_rows,
    train_epoch,
    transform_policy_scores,
)


def average_state_dicts(
    states: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("At least one state dict is required")
    averaged = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            value = torch.zeros_like(first, dtype=torch.float32)
            for state in states:
                value.add_(state[key].float(), alpha=1.0 / len(states))
            averaged[key] = value.to(first.dtype)
        else:
            averaged[key] = first.clone()
    return averaged


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--sweep-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--row-batch-size", type=int, default=8)
    parser.add_argument("--sequence-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep = json.loads(args.sweep_report.read_text())
    selected = sweep["selected"]
    model_name = sweep["model_name"]
    training_rows = load_jsonl(args.input)
    validation_rows = load_jsonl(args.validation_input)
    rows = training_rows + validation_rows
    splits = split_row_ids(training_rows, validation_rows)
    fit_rows = target_training_rows(
        rows, sorted(splits["train"] + splits["internal_test"]), selected["target"]
    )
    evaluation_splits = dict(splits)
    evaluation_splits["internal_test"] = []
    slices = row_slices(rows)
    targets = candidate_targets(rows, selected["target"])
    fit_candidates = candidate_indices(slices, fit_rows)
    target_scale = max(float(np.nanstd(targets[fit_candidates])), 1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    states = []
    seed_reports = []
    tokenizer = None

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model, tokenizer = load_model(model_name, device)
        trainable_parameters = freeze_bottom_layers(
            model, int(selected["frozen_bottom_layers"])
        )
        dataset = RowDataset(rows, fit_rows)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset, batch_size=args.row_batch_size, shuffle=True,
            collate_fn=collate_rows, num_workers=0, generator=generator,
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(selected["learning_rate"]),
            weight_decay=float(sweep.get("weight_decay", 0.01)),
        )
        epochs = int(selected["epoch"])
        scheduler = linear_schedule(optimizer, epochs * len(loader))
        weights = group_weights(rows, fit_rows)
        training = []
        for _ in range(epochs):
            training.append(train_epoch(
                model, tokenizer, loader, optimizer, scheduler,
                device=device, target_name=selected["target"],
                target_scale=target_scale,
                minimum_gain=float(sweep["minimum_gain"]),
                regression_weight=float(sweep["regression_weight"]),
                max_length=args.max_length,
                sequence_batch_size=args.sequence_batch_size,
                use_bf16=use_bf16, weights=weights,
                gradient_clip=float(sweep.get("gradient_clip", 1.0)),
                loss_mode=selected["loss_mode"],
                listwise_temperature=float(sweep["listwise_temperature"]),
                query_mode=selected.get("query_mode", "full"),
            ))
        predictions, predicted_slices, timing = predict_rows(
            model, tokenizer, rows, device=device, max_length=args.max_length,
            row_batch_size=args.row_batch_size,
            sequence_batch_size=args.sequence_batch_size, use_bf16=use_bf16,
            query_mode=selected.get("query_mode", "full"),
        )
        report = evaluate_checkpoint(
            rows, targets, predictions, predicted_slices, evaluation_splits,
            minimum_gain=float(sweep["minimum_gain"]),
            minimum_precision=float(sweep["minimum_precision"]),
            minimum_emitted=int(sweep.get("minimum_emitted", 5)),
            score_mode=selected.get("score_mode", "margin"),
        )
        seed_reports.append({
            "seed": seed, "training": training, "timing": timing,
            "trainable_parameters": trainable_parameters,
            "evaluation": report,
        })
        states.append(cpu_state_dict(model))
        del model, optimizer, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model, tokenizer = load_model(model_name, device)
    model.load_state_dict(average_state_dicts(states), strict=True)
    predictions, predicted_slices, timing = predict_rows(
        model, tokenizer, rows, device=device, max_length=args.max_length,
        row_batch_size=args.row_batch_size,
        sequence_batch_size=args.sequence_batch_size, use_bf16=use_bf16,
        query_mode=selected.get("query_mode", "full"),
    )
    soup_report = evaluate_checkpoint(
        rows, targets, predictions, predicted_slices, evaluation_splits,
        minimum_gain=float(sweep["minimum_gain"]),
        minimum_precision=float(sweep["minimum_precision"]),
        minimum_emitted=int(sweep.get("minimum_emitted", 5)),
        score_mode=selected.get("score_mode", "margin"),
    )
    metadata = selected | {
        "stage": "refit_seed_soup",
        "fit_rows": len(fit_rows),
        "seeds": args.seeds,
        "threshold": soup_report["threshold"],
        "target_scale": target_scale,
    }
    model_bytes = save_best(model, tokenizer, args.output_dir, metadata)
    np.savez_compressed(
        args.output_dir / "selected_predictions.npz",
        predictions=predictions.astype(np.float16),
        policy_predictions=transform_policy_scores(
            predictions, predicted_slices, selected.get("score_mode", "margin")
        ).astype(np.float16),
        row_slices=np.asarray(predicted_slices, dtype=np.int32),
        threshold=np.asarray([soup_report["threshold"]], dtype=np.float32),
    )
    report = {
        "model_name": model_name,
        "source_sweep": args.sweep_report.resolve().as_posix(),
        "selected_configuration": selected,
        "fit_rows": len(fit_rows),
        "fit_unique_groups": len({rows[index]["question_group"] for index in fit_rows}),
        "calibration_rows": len(splits["calibration"]),
        "seeds": args.seeds,
        "target_scale": target_scale,
        "individual_seeds": seed_reports,
        "soup": soup_report | {"timing": timing},
        "model_bytes": model_bytes,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "selected_configuration": selected,
        "fit_rows": len(fit_rows),
        "soup_calibration": soup_report["calibration"],
        "soup_frozen_validation": soup_report["frozen_validation"],
        "soup_frozen_novel": soup_report["frozen_novel"],
        "model_bytes": model_bytes,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
