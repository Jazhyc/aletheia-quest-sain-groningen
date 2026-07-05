#!/usr/bin/env python3
"""Fine-tune MiniLM as a black-box deception detector.

The training protocol mirrors the lightweight text probes:
fit candidates on the public train split only, select hyperparameters plus the
binary threshold on validation, and optionally report the held-out local test
split once for confirmation.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import importlib.metadata
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.text_probe.run_text_probe import (  # noqa: E402
    SplitData,
    load_split,
    macro_metrics,
    per_dataset_table,
    select_threshold,
    write_predictions,
)


DEFAULT_MODEL_ID = "microsoft/MiniLM-L12-H384-uncased"


@dataclasses.dataclass(frozen=True)
class Candidate:
    view: str
    max_length: int
    optimizer: str
    learning_rate: float
    muon_learning_rate: float | None
    epochs: int
    weight_decay: float
    warmup_ratio: float
    class_weights: bool
    seed: int


class TextDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame, view: str) -> None:
        self.texts = frame[view].fillna("").astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"text": self.texts[index], "label": self.labels[index]}


def parse_csv_floats(value: str) -> list[float]:
    out = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    return out


def parse_csv_ints(value: str) -> list[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def parse_csv_strings(value: str) -> list[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one value")
    return out


def parse_csv_bools(value: str) -> list[bool]:
    out = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"1", "true", "yes", "y"}:
            out.append(True)
        elif item in {"0", "false", "no", "n"}:
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(f"expected boolean, got {item!r}")
    if not out:
        raise argparse.ArgumentTypeError("expected at least one boolean")
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def candidate_grid(
    *,
    views: list[str],
    max_lengths: list[int],
    optimizers: list[str],
    learning_rates: list[float],
    muon_learning_rates: list[float],
    epochs_grid: list[int],
    weight_decays: list[float],
    warmup_ratios: list[float],
    class_weight_options: list[bool],
    seeds: list[int],
) -> list[Candidate]:
    valid_views = {"output", "dialogue", "output_context"}
    invalid = sorted(set(views) - valid_views)
    if invalid:
        raise ValueError(f"invalid views: {invalid}")
    valid_optimizers = {"adamw", "muon_adamw"}
    invalid_optimizers = sorted(set(optimizers) - valid_optimizers)
    if invalid_optimizers:
        raise ValueError(f"invalid optimizers: {invalid_optimizers}")

    candidates = []
    for optimizer in optimizers:
        optimizer_muon_lrs: list[float | None]
        if optimizer == "muon_adamw":
            optimizer_muon_lrs = list(muon_learning_rates)
        else:
            optimizer_muon_lrs = [None]
        for view in views:
            for max_length in max_lengths:
                for learning_rate in learning_rates:
                    for muon_learning_rate in optimizer_muon_lrs:
                        for epochs in epochs_grid:
                            for weight_decay in weight_decays:
                                for warmup_ratio in warmup_ratios:
                                    for class_weights_enabled in class_weight_options:
                                        for seed in seeds:
                                            candidates.append(Candidate(
                                                view=view,
                                                max_length=max_length,
                                                optimizer=optimizer,
                                                learning_rate=learning_rate,
                                                muon_learning_rate=muon_learning_rate,
                                                epochs=epochs,
                                                weight_decay=weight_decay,
                                                warmup_ratio=warmup_ratio,
                                                class_weights=class_weights_enabled,
                                                seed=seed,
                                            ))
    return candidates


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=2).astype("float32")
    if np.any(counts == 0):
        return torch.ones(2, device=device)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_collate_fn(tokenizer: Any, *, max_length: int) -> Any:
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return encoded

    return collate


def optimizer_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in no_decay):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]


def linear_warmup_scale(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        scale = float(step + 1) / float(warmup_steps)
    else:
        remaining = max(total_steps - step - 1, 0)
        decay_steps = max(total_steps - warmup_steps, 1)
        scale = remaining / decay_steps
    return max(scale, 0.0)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate ``UV^T`` for matrix ``USV^T`` using quintic Newton-Schulz."""
    assert matrix.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = matrix.bfloat16()
    if matrix.size(0) > matrix.size(1):
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x
    if matrix.size(0) > matrix.size(1):
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D weight matrices.

    Use AdamW for biases, normalization weights, embeddings, and classifier
    heads; Muon's orthogonalized updates are intended for hidden matrices.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        *,
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.ndim != 2:
                    raise RuntimeError("Muon received a non-2D parameter")
                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(grad.float())
                update = grad.float().add(buffer, alpha=momentum) if nesterov else buffer
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                scale = max(1.0, param.size(0) / param.size(1)) ** 0.5
                param.add_(update.to(param.dtype), alpha=-lr * scale)
        return loss


def split_muon_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[tuple[str, torch.nn.Parameter]]]:
    muon_params = []
    adamw_params = []
    excluded = ("embeddings", "classifier", "pooler")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and not any(token in name for token in excluded):
            muon_params.append(param)
        else:
            adamw_params.append((name, param))
    return muon_params, adamw_params


def build_optimizers(model: torch.nn.Module, candidate: Candidate) -> list[tuple[torch.optim.Optimizer, float]]:
    if candidate.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_groups(model, candidate.weight_decay),
            lr=candidate.learning_rate,
        )
        return [(optimizer, candidate.learning_rate)]

    if candidate.optimizer == "muon_adamw":
        if candidate.muon_learning_rate is None:
            raise ValueError("muon_adamw requires muon_learning_rate")
        muon_params, adamw_named_params = split_muon_parameters(model)
        optimizers: list[tuple[torch.optim.Optimizer, float]] = []
        if muon_params:
            optimizers.append((
                Muon(
                    muon_params,
                    lr=candidate.muon_learning_rate,
                    weight_decay=candidate.weight_decay,
                ),
                candidate.muon_learning_rate,
            ))
        if adamw_named_params:
            no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
            decay_params = []
            nodecay_params = []
            for name, param in adamw_named_params:
                if any(token in name for token in no_decay):
                    nodecay_params.append(param)
                else:
                    decay_params.append(param)
            optimizers.append((
                torch.optim.AdamW(
                    [
                        {"params": decay_params, "weight_decay": candidate.weight_decay},
                        {"params": nodecay_params, "weight_decay": 0.0},
                    ],
                    lr=candidate.learning_rate,
                ),
                candidate.learning_rate,
            ))
        return optimizers

    raise ValueError(f"unknown optimizer={candidate.optimizer!r}")


def score_model(
    model: torch.nn.Module,
    tokenizer: Any,
    data: SplitData,
    *,
    view: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    use_bf16: bool,
) -> pd.DataFrame:
    dataset = TextDataset(data.frame, view)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(tokenizer, max_length=max_length),
    )
    scores: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            del labels
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**batch).logits
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            scores.append(probs.cpu().numpy())
    out = data.frame[["dataset", "index", "label"]].copy()
    out["score"] = np.concatenate(scores, axis=0)
    return out


def best_key(metrics: dict[str, float | None]) -> tuple[float, float, float]:
    return (
        metrics["balanced_accuracy"] if metrics["balanced_accuracy"] is not None else -1.0,
        metrics["auroc"] if metrics["auroc"] is not None else -1.0,
        -(metrics["fpr"] if metrics["fpr"] is not None else 1.0),
    )


def train_candidate(
    candidate: Candidate,
    *,
    model_id: str,
    train: SplitData,
    validation: SplitData,
    tokenizer: Any,
    device: torch.device,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    use_bf16: bool,
) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification

    set_seed(candidate.seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    model.to(device)

    labels = train.frame["label"].to_numpy()
    weights = class_weights(labels, device) if candidate.class_weights else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizers = build_optimizers(model, candidate)

    train_dataset = TextDataset(train.frame, candidate.view)
    generator = torch.Generator()
    generator.manual_seed(candidate.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=build_collate_fn(tokenizer, max_length=candidate.max_length),
    )

    updates_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_steps = max(candidate.epochs * updates_per_epoch, 1)
    warmup_steps = int(total_steps * candidate.warmup_ratio)
    global_step = 0
    best: dict[str, Any] | None = None

    for epoch in range(1, candidate.epochs + 1):
        model.train()
        for optimizer, _ in optimizers:
            optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels_tensor = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**batch).logits
                loss = criterion(logits.float(), labels_tensor)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            total_loss += float(loss.detach().cpu()) * gradient_accumulation_steps

            should_step = (
                batch_index % gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr_scale = linear_warmup_scale(
                    step=global_step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                )
                for optimizer, base_lr in optimizers:
                    set_lr(optimizer, base_lr * lr_scale)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                global_step += 1

        validation_scores = score_model(
            model,
            tokenizer,
            validation,
            view=candidate.view,
            max_length=candidate.max_length,
            batch_size=eval_batch_size,
            device=device,
            use_bf16=use_bf16,
        )
        threshold, metrics = select_threshold(validation_scores)
        epoch_row = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "threshold": threshold,
            **metrics,
        }
        print(json.dumps({"candidate": dataclasses.asdict(candidate), **epoch_row}), flush=True)
        if best is None or best_key(metrics) > best_key(best["metrics"]):
            best = {
                "epoch": epoch,
                "threshold": threshold,
                "metrics": metrics,
                "validation_scores": validation_scores,
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "train_loss": epoch_row["train_loss"],
            }

    if best is None:
        raise RuntimeError("candidate produced no validation result")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best


def load_tokenizer(model_id: str) -> Any:
    importlib.metadata.packages_distributions = lambda: {}
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def save_model(output_dir: Path, model: torch.nn.Module, tokenizer: Any) -> None:
    model_dir = output_dir / "best_model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def load_model_from_state(
    state: dict[str, torch.Tensor],
    *,
    model_id: str,
    device: torch.device,
) -> torch.nn.Module:
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    model.load_state_dict(state)
    model.to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "blackbox" / "minilm_finetune_v1")
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--views", type=parse_csv_strings, default=["output_context"])
    parser.add_argument("--max-lengths", type=parse_csv_ints, default=[256, 384])
    parser.add_argument("--optimizers", type=parse_csv_strings, default=["adamw"])
    parser.add_argument("--learning-rates", type=parse_csv_floats, default=[1e-5, 2e-5, 4e-5])
    parser.add_argument("--muon-learning-rates", type=parse_csv_floats, default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--epochs-grid", type=parse_csv_ints, default=[2, 3])
    parser.add_argument("--weight-decays", type=parse_csv_floats, default=[0.01])
    parser.add_argument("--warmup-ratios", type=parse_csv_floats, default=[0.06])
    parser.add_argument("--class-weight-options", type=parse_csv_bools, default=[True])
    parser.add_argument("--seeds", type=parse_csv_ints, default=[0])
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    if args.no_class_weights:
        args.class_weight_options = [False]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_bf16 = device.type == "cuda" and not args.no_bf16
    print(f"device={device} bf16={use_bf16} model_id={args.model_id}", flush=True)

    print("loading train split", flush=True)
    train = load_split("train", args.splits_dir, max_context_chars=args.max_context_chars)
    print(f"train rows={len(train.frame)} datasets={len(train.datasets)} positives={int(train.frame['label'].sum())}", flush=True)
    print("loading validation split", flush=True)
    validation = load_split("validation", args.splits_dir, max_context_chars=args.max_context_chars)
    print(f"validation rows={len(validation.frame)} datasets={len(validation.datasets)} positives={int(validation.frame['label'].sum())}", flush=True)

    tokenizer = load_tokenizer(args.model_id)
    candidates = candidate_grid(
        views=args.views,
        max_lengths=args.max_lengths,
        optimizers=args.optimizers,
        learning_rates=args.learning_rates,
        muon_learning_rates=args.muon_learning_rates,
        epochs_grid=args.epochs_grid,
        weight_decays=args.weight_decays,
        warmup_ratios=args.warmup_ratios,
        class_weight_options=args.class_weight_options,
        seeds=args.seeds,
    )

    grid_rows = []
    best: dict[str, Any] | None = None
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {candidate}", flush=True)
        trained = train_candidate(
            candidate,
            model_id=args.model_id,
            train=train,
            validation=validation,
            tokenizer=tokenizer,
            device=device,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_bf16=use_bf16,
        )
        row = {
            **dataclasses.asdict(candidate),
            "selected_epoch": trained["epoch"],
            "train_loss": trained["train_loss"],
            "threshold": trained["threshold"],
            **trained["metrics"],
        }
        grid_rows.append(row)
        current = best_key(trained["metrics"])
        previous = best_key(best["validation"]["metrics"]) if best is not None else (-1.0, -1.0, -1.0)
        if best is None or current > previous:
            best = {
                "candidate": dataclasses.asdict(candidate),
                "selected_epoch": trained["epoch"],
                "threshold": float(trained["threshold"]),
                "model_state": trained["model_state"],
                "validation_scores": trained["validation_scores"],
                "validation": {
                    "metrics": trained["metrics"],
                    "datasets": per_dataset_table(trained["validation_scores"], trained["threshold"]),
                },
            }
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError("no candidates evaluated")

    pd.DataFrame(grid_rows).sort_values(
        ["balanced_accuracy", "auroc", "fpr"],
        ascending=[False, False, True],
    ).to_csv(args.output_dir / "validation_grid.csv", index=False)

    threshold = float(best["threshold"])
    write_predictions(args.output_dir / "predictions" / "validation.csv", best["validation_scores"], threshold)
    selected_model = load_model_from_state(
        best["model_state"],
        model_id=args.model_id,
        device=device,
    )
    save_model(args.output_dir, selected_model, tokenizer)

    selected = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "MiniLM sequence classifier; fit train, select hyperparameters/epoch/threshold on validation",
        "model_id": args.model_id,
        "bf16": use_bf16,
        "candidate": best["candidate"],
        "selected_epoch": best["selected_epoch"],
        "threshold": threshold,
        "train": {
            "rows": int(len(train.frame)),
            "datasets": train.datasets,
            "positives": int(train.frame["label"].sum()),
        },
        "validation": best["validation"],
    }

    if args.include_test:
        print("loading test split", flush=True)
        test = load_split("test", args.splits_dir, max_context_chars=args.max_context_chars)
        test_scores = score_model(
            selected_model,
            tokenizer,
            test,
            view=best["candidate"]["view"],
            max_length=best["candidate"]["max_length"],
            batch_size=args.eval_batch_size,
            device=device,
            use_bf16=use_bf16,
        )
        selected["test"] = {
            "rows": int(len(test.frame)),
            "datasets": test.datasets,
            "positives": int(test.frame["label"].sum()),
            "metrics": macro_metrics(test_scores, threshold),
            "datasets_table": per_dataset_table(test_scores, threshold),
        }
        write_predictions(args.output_dir / "predictions" / "test.csv", test_scores, threshold)

    result_for_json = selected
    (args.output_dir / "result.json").write_text(json.dumps(result_for_json, indent=2) + "\n")
    print(json.dumps(result_for_json, indent=2), flush=True)


if __name__ == "__main__":
    main()
