#!/usr/bin/env python3
"""Score PID examples with a first-order DataRater meta-gradient proxy.

For one virtual gradient step, the derivative of held-out loss with respect to
an example weight is proportional to the negative dot product between that
example's gradient and the held-out gradient.  This script measures that
alignment in a small LoRA subspace, then writes balanced top-k manifests for
matched downstream SFT runs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.train_student_sft import (
    load_records,
    tokenize_record,
)


RecordKey = tuple[str, Any]


def record_key(record: dict[str, Any]) -> RecordKey:
    """Return the stable dataset/index identity for a teacher record."""
    return str(record["dataset"]), record["index"]


def record_stratum(record: dict[str, Any]) -> tuple[str, int]:
    """Return the dataset/label stratum used by every selection operation."""
    return str(record["dataset"]), int(record["label"])


def stable_digest(record: dict[str, Any], seed: int, namespace: str) -> bytes:
    """Hash a record deterministically without depending on input order."""
    dataset, index = record_key(record)
    label = int(record["label"])
    return hashlib.sha256(
        f"{namespace}\0{seed}\0{dataset}\0{label}\0{index}".encode("utf-8")
    ).digest()


def split_meta_records(
    records: Sequence[dict[str, Any]],
    fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reserve a stable, balanced meta-objective fraction from each stratum."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("meta fraction must be in (0, 1)")
    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[record_stratum(record)].append(record)

    meta_keys: set[RecordKey] = set()
    for stratum_records in strata.values():
        count = max(1, int(len(stratum_records) * fraction + 0.5))
        ordered = sorted(
            stratum_records,
            key=lambda record: stable_digest(record, seed, "meta"),
        )
        meta_keys.update(record_key(record) for record in ordered[:count])

    meta = [record for record in records if record_key(record) in meta_keys]
    candidates = [record for record in records if record_key(record) not in meta_keys]
    return meta, candidates


def limit_records_balanced(
    records: Sequence[dict[str, Any]],
    limit: int | None,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    """Take an approximately balanced deterministic subset for GPU smokes."""
    if limit is None or limit >= len(records):
        return list(records)
    if limit <= 0:
        raise ValueError("record limit must be positive")

    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[record_stratum(record)].append(record)
    for stratum in strata:
        strata[stratum].sort(
            key=lambda record: stable_digest(record, seed, namespace)
        )

    selected: list[dict[str, Any]] = []
    ordered_strata = sorted(strata)
    cursor = Counter()
    while len(selected) < limit:
        made_progress = False
        for stratum in ordered_strata:
            position = cursor[stratum]
            if position < len(strata[stratum]):
                selected.append(strata[stratum][position])
                cursor[stratum] += 1
                made_progress = True
                if len(selected) == limit:
                    break
        if not made_progress:
            break
    return selected


def select_scored_fraction(
    records: Sequence[dict[str, Any]],
    scores: dict[RecordKey, float],
    keep_fraction: float,
    seed: int,
    *,
    highest: bool = True,
) -> list[dict[str, Any]]:
    """Select a score-ranked fraction independently in every stratum."""
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep fraction must be in (0, 1]")
    missing = [record_key(record) for record in records if record_key(record) not in scores]
    if missing:
        raise ValueError(f"missing scores for {len(missing)} records")

    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[record_stratum(record)].append(record)

    selected_keys: set[RecordKey] = set()
    direction = -1.0 if highest else 1.0
    for stratum_records in strata.values():
        count = max(1, int(len(stratum_records) * keep_fraction + 0.5))
        ordered = sorted(
            stratum_records,
            key=lambda record: (
                direction * scores[record_key(record)],
                stable_digest(record, seed, "score-tie"),
            ),
        )
        selected_keys.update(record_key(record) for record in ordered[:count])
    return [record for record in records if record_key(record) in selected_keys]


def select_random_fraction(
    records: Sequence[dict[str, Any]],
    keep_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Select the matched random control within every stratum."""
    random_scores = {
        record_key(record): int.from_bytes(
            stable_digest(record, seed, "random-control")[:8], "big"
        )
        / 2**64
        for record in records
    }
    return select_scored_fraction(
        records,
        random_scores,
        keep_fraction,
        seed,
        highest=True,
    )


def manifest_row(record: dict[str, Any]) -> dict[str, Any]:
    """Return the strict manifest schema accepted by the student trainer."""
    return {
        "dataset": str(record["dataset"]),
        "index": record["index"],
        "label": int(record["label"]),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def per_sequence_completion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return mean completion-token cross entropy for each batch row."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    token_losses = F.cross_entropy(
        shift_logits.transpose(1, 2),
        shift_labels,
        reduction="none",
        ignore_index=-100,
    )
    mask = shift_labels.ne(-100)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("each record must contain at least one supervised token")
    return (token_losses * mask).sum(dim=1) / counts


def select_lora_parameters(
    model: torch.nn.Module,
    last_layers: int,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Keep only LoRA parameters from the final transformer layers trainable."""
    layer_count = int(model.config.num_hidden_layers)
    if not 1 <= last_layers <= layer_count:
        raise ValueError(f"last_layers must be in [1, {layer_count}]")
    first_layer = layer_count - last_layers

    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        in_selected_layer = any(
            f".layers.{layer_index}." in name
            for layer_index in range(first_layer, layer_count)
        )
        keep = "lora_" in name and in_selected_layer
        parameter.requires_grad_(keep)
        if keep:
            selected.append((name, parameter))
    if not selected:
        raise RuntimeError(
            "no LoRA parameters matched the requested final transformer layers"
        )
    return selected


def record_tensors(
    record: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    device: torch.device,
    *,
    prediction_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize one record and move its tensors to the scoring device."""
    feature = tokenize_record(
        record,
        tokenizer,
        max_length,
        target_mode="prediction_only" if prediction_only else "teacher",
    )
    input_ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([feature["labels"]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, labels


def example_gradient(
    model: torch.nn.Module,
    parameters: Sequence[torch.nn.Parameter],
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[float, list[torch.Tensor]]:
    """Compute a single sequence loss and its selected-parameter gradients."""
    input_ids, attention_mask, labels = tensors
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = per_sequence_completion_loss(outputs.logits, labels).mean()
    gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    return float(loss.detach()), [gradient.detach().float() for gradient in gradients]


def gradient_alignment(
    gradients: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
    reference_norm: float,
) -> tuple[float, float, float]:
    """Return dot product, cosine similarity, and example gradient norm."""
    dot = sum(
        float(torch.sum(gradient * meta_gradient))
        for gradient, meta_gradient in zip(gradients, reference, strict=True)
    )
    norm_sq = sum(float(torch.sum(gradient * gradient)) for gradient in gradients)
    norm = math.sqrt(max(norm_sq, 0.0))
    cosine = dot / (norm * reference_norm) if norm > 0.0 and reference_norm > 0.0 else 0.0
    return dot, cosine, norm


def parse_fraction_list(raw: str) -> list[float]:
    """Parse comma-separated keep fractions."""
    values = [float(value) for value in raw.split(",") if value.strip()]
    if not values or any(not 0.0 < value <= 1.0 for value in values):
        raise ValueError("keep fractions must be comma-separated values in (0, 1]")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/"
            "teacher/train.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name-contains", default="varied-deception")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--meta-fraction", type=float, default=0.05)
    parser.add_argument("--keep-fractions", default="0.25,0.5,0.75")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--last-layers", type=int, default=1)
    parser.add_argument("--max-meta-records", type=int)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    records = load_records(
        args.input,
        dataset_name_contains=args.dataset_name_contains,
    )
    meta_records, candidate_records = split_meta_records(
        records,
        args.meta_fraction,
        args.seed,
    )
    meta_records = limit_records_balanced(
        meta_records,
        args.max_meta_records,
        args.seed,
        "meta-limit",
    )
    candidate_records = limit_records_balanced(
        candidate_records,
        args.max_candidates,
        args.seed,
        "candidate-limit",
    )
    print(
        f"records={len(records)} meta={len(meta_records)} "
        f"candidates={len(candidate_records)}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("DataRater gradient scoring requires a CUDA GPU")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    model.to(device)
    model.eval()
    named_parameters = select_lora_parameters(model, args.last_layers)
    parameters = [parameter for _, parameter in named_parameters]
    parameter_count = sum(parameter.numel() for parameter in parameters)
    print(
        f"selected_lora_parameters={len(parameters)} "
        f"selected_lora_parameter_count={parameter_count}"
    )

    meta_gradient = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters]
    meta_losses: list[float] = []
    for position, record in enumerate(meta_records, start=1):
        loss, gradients = example_gradient(
            model,
            parameters,
            record_tensors(
                record,
                tokenizer,
                args.max_length,
                device,
                prediction_only=True,
            ),
        )
        meta_losses.append(loss)
        for accumulator, gradient in zip(meta_gradient, gradients, strict=True):
            accumulator.add_(gradient / len(meta_records))
        if position % args.log_every == 0 or position == len(meta_records):
            print(f"meta {position}/{len(meta_records)} mean_loss={sum(meta_losses)/len(meta_losses):.6f}")

    reference_norm = math.sqrt(
        sum(float(torch.sum(gradient * gradient)) for gradient in meta_gradient)
    )
    if not math.isfinite(reference_norm) or reference_norm == 0.0:
        raise RuntimeError(f"invalid meta-gradient norm: {reference_norm}")
    print(f"meta_gradient_norm={reference_norm:.8f}")

    scored_rows: list[dict[str, Any]] = []
    for position, record in enumerate(candidate_records, start=1):
        loss, gradients = example_gradient(
            model,
            parameters,
            record_tensors(
                record,
                tokenizer,
                args.max_length,
                device,
                prediction_only=False,
            ),
        )
        dot, cosine, gradient_norm = gradient_alignment(
            gradients,
            meta_gradient,
            reference_norm,
        )
        scored_rows.append(
            {
                **manifest_row(record),
                "completion_loss": loss,
                "gradient_dot": dot,
                "gradient_cosine": cosine,
                "gradient_norm": gradient_norm,
            }
        )
        if position % args.log_every == 0 or position == len(candidate_records):
            recent = scored_rows[-min(args.log_every, len(scored_rows)) :]
            mean_cosine = sum(row["gradient_cosine"] for row in recent) / len(recent)
            print(
                f"candidate {position}/{len(candidate_records)} "
                f"recent_mean_cosine={mean_cosine:.6f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "scores.jsonl", scored_rows)
    write_jsonl(args.output_dir / "meta_manifest.jsonl", map(manifest_row, meta_records))

    score_maps = {
        "dot": {
            (row["dataset"], row["index"]): float(row["gradient_dot"])
            for row in scored_rows
        },
        "cosine": {
            (row["dataset"], row["index"]): float(row["gradient_cosine"])
            for row in scored_rows
        },
        "loss": {
            (row["dataset"], row["index"]): float(row["completion_loss"])
            for row in scored_rows
        },
    }
    manifest_counts: dict[str, int] = {}
    for keep_fraction in parse_fraction_list(args.keep_fractions):
        suffix = f"keep{int(round(100 * keep_fraction)):02d}"
        random_records = select_random_fraction(
            candidate_records,
            keep_fraction,
            args.seed,
        )
        random_name = f"random_{suffix}"
        write_jsonl(
            args.output_dir / "manifests" / f"{random_name}.jsonl",
            map(manifest_row, random_records),
        )
        manifest_counts[random_name] = len(random_records)
        for metric, scores in score_maps.items():
            selected = select_scored_fraction(
                candidate_records,
                scores,
                keep_fraction,
                args.seed,
            )
            name = f"{metric}_{suffix}"
            write_jsonl(
                args.output_dir / "manifests" / f"{name}.jsonl",
                map(manifest_row, selected),
            )
            manifest_counts[name] = len(selected)

    summary = {
        "input": args.input.as_posix(),
        "dataset_name_contains": args.dataset_name_contains,
        "seed": args.seed,
        "meta_fraction": args.meta_fraction,
        "meta_records": len(meta_records),
        "candidate_records": len(candidate_records),
        "meta_loss_mean": sum(meta_losses) / len(meta_losses),
        "meta_gradient_norm": reference_norm,
        "selected_lora_parameters": [name for name, _ in named_parameters],
        "selected_lora_parameter_count": parameter_count,
        "manifest_counts": manifest_counts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
