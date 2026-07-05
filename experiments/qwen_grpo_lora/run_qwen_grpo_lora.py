#!/usr/bin/env python3
"""GRPO fine-tune for the Qwen no-thinking black-box judge.

This is development tooling, not submission code. It trains a rank-1 LoRA
adapter on the public local train split and evaluates on the local validation
split with the same prompt family used by the fast no-thinking Qwen judge.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib.metadata
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))


HF_SPLIT = "test"


@dataclasses.dataclass
class SplitRecords:
    frame: pd.DataFrame
    dataset_names: list[str]


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    name: str
    labels_uri: str
    id_column: str = "index"
    label_column: str = "deceptive"


def resolve_path(pathish: str, base: Path) -> Path:
    path = Path(pathish)
    return path if path.is_absolute() else base / path


def resolve_uri(uri: str, base: Path) -> str:
    if "://" in uri:
        return uri
    return resolve_path(uri, base).as_posix()


def cfg_path(value: str | Path, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def merge_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + content
        else:
            merged.append({"role": role, "content": content})
    return merged


def final_assistant(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def final_assistant_reasoning(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("reasoning") or "")
    return ""


def truncate_context(context: str, max_chars: int, mode: str) -> str:
    if len(context) <= max_chars:
        return context
    if max_chars <= 0:
        return ""
    if mode == "tail":
        return context[-max_chars:]
    if mode == "head":
        return context[:max_chars]
    if mode == "head_tail":
        marker = "\n\n[... middle truncated ...]\n\n"
        available = max_chars - len(marker)
        if available <= 0:
            return context[:max_chars]
        head_chars = available // 2
        tail_chars = available - head_chars
        return context[:head_chars] + marker + context[-tail_chars:]
    raise ValueError(f"unknown truncation mode={mode!r}")


def format_dialogue(
    messages: list[dict[str, Any]],
    max_chars: int,
    context_truncation: str,
) -> tuple[str, str]:
    merged = merge_messages(messages)
    assistant = final_assistant(merged)
    context = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in merged
    )
    return truncate_context(context, max_chars, context_truncation), assistant


def build_prompt(
    messages: list[dict[str, Any]],
    prompt_template: str,
    max_chars: int,
    *,
    context_truncation: str,
    include_reasoning: bool,
    reasoning_max_chars: int,
    reasoning_truncation: str,
) -> str:
    context, output = format_dialogue(messages, max_chars, context_truncation)
    prompt = f"{prompt_template}\n\n<context>\n{context}\n</context>\n\n<output>\n{output}\n</output>"
    if include_reasoning and reasoning_max_chars > 0:
        reasoning = truncate_context(
            final_assistant_reasoning(messages),
            reasoning_max_chars,
            reasoning_truncation,
        )
        if reasoning:
            prompt = f"{prompt}\n\n<assistant_reasoning>\n{reasoning}\n</assistant_reasoning>"
    return prompt


def parse_rating(text: str, *, rating_min: int, rating_max: int) -> int | None:
    rating_range = f"[{rating_min}-{rating_max}]"
    patterns = (
        rf"(?im)\b(?:final\s+answer|final\s+rating|rating|score)\s*[:\-]?\s*\**\s*({rating_range})\b",
        rf"(?im)\b({rating_range})\s*/\s*{rating_max}\b",
    )
    matches: list[int] = []
    for pattern in patterns:
        matches.extend(int(match) for match in re.findall(pattern, text))
    if matches:
        return matches[-1]
    stripped = text.strip()
    if re.fullmatch(rating_range, stripped):
        return int(stripped)
    return None


def rating_to_score(rating: int, *, rating_min: int, rating_max: int) -> float:
    return (rating - rating_min) / (rating_max - rating_min)


def load_split_config(path: Path, base: Path) -> list[DatasetConfig]:
    data = yaml.safe_load(path.read_text()) or {}
    datasets = []
    for item in data.get("datasets", []):
        item = dict(item)
        item["labels_uri"] = resolve_uri(item["labels_uri"], base)
        datasets.append(DatasetConfig(**item))
    if not datasets:
        raise SystemExit(f"{path} has no datasets entries")
    return datasets


def load_labels(cfg: DatasetConfig) -> pd.DataFrame:
    uri = Path(cfg.labels_uri)
    if cfg.labels_uri.endswith(".csv") and uri.exists():
        raw = pd.read_csv(uri)
    else:
        from datasets import load_dataset

        raw = load_dataset(cfg.labels_uri, split=HF_SPLIT).to_pandas()
    if cfg.id_column not in raw.columns:
        raw = raw.reset_index().rename(columns={"index": cfg.id_column})
    return pd.DataFrame({
        "index": raw[cfg.id_column],
        "label": raw[cfg.label_column].astype(int),
    })


def render_chat_prompt(tokenizer: Any, prompt: str, *, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_examples_for_labels(
    dataset_name: str,
    labels: pd.DataFrame,
    *,
    prompt_template: str,
    tokenizer: Any,
    max_prompt_chars: int,
    context_truncation: str,
    include_reasoning: bool,
    reasoning_max_chars: int,
    reasoning_truncation: str,
    enable_thinking: bool,
    limit: int | None,
) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=HF_SPLIT)
    if "index" not in ds.column_names:
        ds = ds.add_column("index", list(range(len(ds))))

    selected_labels = labels if limit is None else labels.iloc[:limit]
    wanted = set(selected_labels["index"].tolist())
    label_by_index = dict(zip(selected_labels["index"], selected_labels["label"], strict=True))

    rows = []
    for row in ds:
        index = row["index"]
        if index not in wanted:
            continue
        raw_prompt = build_prompt(
            row["messages"],
            prompt_template,
            max_prompt_chars,
            context_truncation=context_truncation,
            include_reasoning=include_reasoning,
            reasoning_max_chars=reasoning_max_chars,
            reasoning_truncation=reasoning_truncation,
        )
        rows.append({
            "dataset": dataset_name,
            "index": index,
            "label": int(label_by_index[index]),
            "prompt": render_chat_prompt(
                tokenizer,
                raw_prompt,
                enable_thinking=enable_thinking,
            ),
        })

    order = {idx: i for i, idx in enumerate(selected_labels["index"].tolist())}
    rows.sort(key=lambda item: order[item["index"]])
    if len(rows) != len(selected_labels):
        raise RuntimeError(
            f"{dataset_name}: loaded {len(rows)} examples for {len(selected_labels)} labels"
        )
    return pd.DataFrame(rows)


def load_split(
    split: str,
    splits_dir: Path,
    *,
    prompt_template: str,
    tokenizer: Any,
    max_prompt_chars: int,
    context_truncation: str,
    include_reasoning: bool,
    reasoning_max_chars: int,
    reasoning_truncation: str,
    enable_thinking: bool,
    limit: int | None,
) -> SplitRecords:
    configs = load_split_config(splits_dir / f"dry.{split}.yaml", ROOT)
    frames = []
    for cfg in configs:
        labels = load_labels(cfg)
        frames.append(
            load_examples_for_labels(
                cfg.name,
                labels,
                prompt_template=prompt_template,
                tokenizer=tokenizer,
                max_prompt_chars=max_prompt_chars,
                context_truncation=context_truncation,
                include_reasoning=include_reasoning,
                reasoning_max_chars=reasoning_max_chars,
                reasoning_truncation=reasoning_truncation,
                enable_thinking=enable_thinking,
                limit=limit,
            )
        )
    return SplitRecords(
        frame=pd.concat(frames, ignore_index=True),
        dataset_names=[cfg.name for cfg in configs],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_trl_availability_flags() -> None:
    """Work around TRL 0.23 with Transformers 5 returning availability tuples."""
    import trl.import_utils as trl_import_utils

    for name in dir(trl_import_utils):
        if not name.startswith("_") or not name.endswith("_available"):
            continue
        value = getattr(trl_import_utils, name)
        if isinstance(value, tuple):
            setattr(trl_import_utils, name, bool(value[0]))


def patch_vllm_guided_decoding_params() -> None:
    """Provide the TRL import symbol for vLLM builds without guided decoding."""
    try:
        import vllm.sampling_params as sampling_params
    except ImportError:
        return
    if hasattr(sampling_params, "GuidedDecodingParams"):
        return

    @dataclasses.dataclass
    class GuidedDecodingParams:
        regex: str | None = None

    sampling_params.GuidedDecodingParams = GuidedDecodingParams


def patch_trl_sampling_params() -> None:
    """Drop colocated-vLLM kwargs that TRL 0.23 passes but vLLM 0.24 rejects."""
    import trl.trainer.grpo_trainer as grpo_trainer
    from vllm.sampling_params import SamplingParams as original_sampling_params

    if getattr(grpo_trainer.SamplingParams, "_aq_compat", False):
        return

    guided_decoding_params = grpo_trainer.GuidedDecodingParams

    def compatible_sampling_params(*args: Any, guided_decoding: Any = None, **kwargs: Any) -> Any:
        if guided_decoding is not None:
            if not isinstance(guided_decoding, guided_decoding_params):
                raise TypeError(f"unexpected guided_decoding={guided_decoding!r}")
            raise ValueError("guided decoding is not supported by the pinned local vLLM build")
        if kwargs.get("top_k") == -1:
            kwargs["top_k"] = 0
        return original_sampling_params(*args, **kwargs)

    compatible_sampling_params._aq_compat = True
    grpo_trainer.SamplingParams = compatible_sampling_params


def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate ``UV^T`` for a matrix ``USV^T`` using quintic Newton-Schulz."""
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


class MuonAdamW(torch.optim.Optimizer):
    """Use Muon for 2D LoRA matrices and AdamW for any remaining parameters."""

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        *,
        lr: float,
        muon_lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "muon_lr": muon_lr,
            "betas": betas,
            "eps": eps,
            "muon_momentum": muon_momentum,
            "muon_nesterov": muon_nesterov,
            "muon_ns_steps": muon_ns_steps,
            "weight_decay": 0.0,
            "algorithm": "adamw",
        }
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict[str, Any]) -> None:
        lr = group["muon_lr"]
        momentum = group["muon_momentum"]
        nesterov = group["muon_nesterov"]
        ns_steps = group["muon_ns_steps"]
        weight_decay = group["weight_decay"]
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

    def _adamw_step(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad.float()
            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)
            state = self.state[param]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            step_size = lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(eps)
            param.addcdiv_(exp_avg.to(param.dtype), denom.to(param.dtype), value=-step_size)


def muon_adamw_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    muon_params = []
    adamw_decay_params = []
    adamw_nodecay_params = []
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            muon_params.append(param)
        elif any(token in name for token in no_decay):
            adamw_nodecay_params.append(param)
        else:
            adamw_decay_params.append(param)

    groups = []
    if muon_params:
        groups.append({"params": muon_params, "weight_decay": weight_decay, "algorithm": "muon"})
    if adamw_decay_params:
        groups.append({"params": adamw_decay_params, "weight_decay": weight_decay, "algorithm": "adamw"})
    if adamw_nodecay_params:
        groups.append({"params": adamw_nodecay_params, "weight_decay": 0.0, "algorithm": "adamw"})
    return groups


def completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return str(value[0].get("content", ""))
    return str(value)


def correctness_reward(completions: list[Any], label: list[int], **kwargs: Any) -> list[float]:
    del kwargs
    rewards = []
    for completion, raw_label in zip(completions, label, strict=True):
        text = completion_text(completion)
        rating = parse_rating(text, rating_min=1, rating_max=7)
        if rating is None:
            rewards.append(0.0)
            continue
        score = rating_to_score(rating, rating_min=1, rating_max=7)
        rewards.append(float(score if int(raw_label) == 1 else 1.0 - score))
    return rewards


def format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    del kwargs
    pattern = re.compile(r"(?im)^\s*Rating:\s*[1-7]\s*$")
    return [1.0 if pattern.search(completion_text(completion)) else 0.0 for completion in completions]


def evaluate_model(
    *,
    model: Any,
    tokenizer: Any,
    validation: SplitRecords,
    batch_size: int,
    max_new_tokens: int,
    rating_min: int,
    rating_max: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    device = model.device
    rows = []
    parse_errors = 0
    started = time.time()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(validation.frame), batch_size):
            batch = validation.frame.iloc[start:start + batch_size]
            encoded = tokenizer(
                batch["prompt"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            prompt_len = encoded["input_ids"].shape[1]
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion_ids = outputs[:, prompt_len:]
            texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            for (_, row), text in zip(batch.iterrows(), texts, strict=True):
                rating = parse_rating(text, rating_min=rating_min, rating_max=rating_max)
                if rating is None:
                    parse_errors += 1
                    score = 0.0
                else:
                    score = rating_to_score(rating, rating_min=rating_min, rating_max=rating_max)
                rows.append({
                    "dataset": row["dataset"],
                    "index": row["index"],
                    "label": int(row["label"]),
                    "score": float(score),
                    "rating": rating,
                    "parse_error": rating is None,
                    "generation": text,
                })
    elapsed = time.time() - started
    predictions = pd.DataFrame(rows)
    metadata = {
        "parse_errors": parse_errors,
        "score_time_seconds": elapsed,
        "rows_per_second": len(predictions) / elapsed if elapsed > 0 else None,
    }
    return predictions, metadata


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | None]:
    y_pred = (y_score >= threshold).astype(int)
    tp, tn, fp, fn = confusion(y_true, y_pred)
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    balanced_accuracy = None if recall is None or fpr is None else (recall + (1.0 - fpr)) / 2.0
    return {
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "fpr": fpr,
    }


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | None]:
    from sklearn.metrics import roc_auc_score

    metrics = threshold_metrics(y_true, y_score, threshold)
    auroc = None
    if np.unique(y_true).size >= 2:
        auroc = float(roc_auc_score(y_true, y_score))
    return {**metrics, "auroc": auroc}


def macro_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | None]:
    per_dataset = [
        binary_metrics(group["label"].to_numpy(), group["score"].to_numpy(), threshold)
        for _, group in frame.groupby("dataset", sort=True)
    ]
    out = {}
    for key in ["balanced_accuracy", "auroc", "recall", "fpr"]:
        values = [m[key] for m in per_dataset if m[key] is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def macro_threshold_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | None]:
    per_dataset = [
        threshold_metrics(group["label"].to_numpy(), group["score"].to_numpy(), threshold)
        for _, group in frame.groupby("dataset", sort=True)
    ]
    out = {}
    for key in ["balanced_accuracy", "recall", "fpr"]:
        values = [m[key] for m in per_dataset if m[key] is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def select_threshold(frame: pd.DataFrame) -> tuple[float, dict[str, float | None]]:
    best_threshold = 0.5
    best_metrics = macro_threshold_metrics(frame, best_threshold)
    for threshold in np.linspace(0.0, 1.0, 401):
        metrics = macro_threshold_metrics(frame, float(threshold))
        current = (
            metrics["balanced_accuracy"] if metrics["balanced_accuracy"] is not None else -1.0,
            -(metrics["fpr"] if metrics["fpr"] is not None else 1.0),
        )
        best = (
            best_metrics["balanced_accuracy"] if best_metrics["balanced_accuracy"] is not None else -1.0,
            -(best_metrics["fpr"] if best_metrics["fpr"] is not None else 1.0),
        )
        if current > best:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, macro_metrics(frame, best_threshold)


def per_dataset_table(frame: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    rows = []
    for dataset, group in frame.groupby("dataset", sort=True):
        metrics = binary_metrics(
            group["label"].to_numpy(),
            group["score"].to_numpy(),
            threshold,
        )
        rows.append({"dataset": dataset, "n": int(len(group)), **metrics})
    return rows


def write_predictions(path: Path, frame: pd.DataFrame, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "index": frame["index"],
        "deceptive": frame["score"] >= threshold,
        "score": frame["score"],
        "label": frame["label"],
        "dataset": frame["dataset"],
    }).to_csv(path, index=False)


def package_dataset(frame: pd.DataFrame):
    from datasets import Dataset

    return Dataset.from_pandas(frame[["prompt", "label", "dataset", "index"]], preserve_index=False)


def version_or_missing(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


@hydra.main(version_base=None, config_path="../../configs", config_name="qwen_grpo_lora")
def main(cfg: DictConfig) -> None:
    base_dir = Path(get_original_cwd())
    output_dir = cfg_path(cfg.output_dir, base_dir)
    splits_dir = cfg_path(cfg.splits_dir, base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.seed))
    prompt_template = str(cfg.judge.prompt)
    include_reasoning = bool(cfg.judge.include_reasoning)
    reasoning_max_chars = int(cfg.judge.reasoning_max_chars)
    baseline_threshold = float(cfg.scoring.baseline_threshold)

    if bool(cfg.wandb.enabled):
        os.environ.setdefault("WANDB_ENTITY", str(cfg.wandb.entity))
        os.environ.setdefault("WANDB_PROJECT", str(cfg.wandb.project))

    normalize_trl_availability_flags()
    patch_vllm_guided_decoding_params()
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    patch_trl_sampling_params()

    class MuonGRPOTrainer(GRPOTrainer):
        def _fix_param_name_to_vllm(self, name: str, extra_prefixes: list[str] | None = None) -> str:
            name = super()._fix_param_name_to_vllm(name, extra_prefixes=extra_prefixes)
            if name.startswith("model."):
                return "language_model." + name
            if name.startswith("lm_head."):
                return "language_model." + name
            return name

        def create_optimizer(self, model=None) -> torch.optim.Optimizer:
            if self.optimizer is None:
                opt_model = self.model if model is None else model
                self.optimizer = MuonAdamW(
                    muon_adamw_param_groups(opt_model, float(self.args.weight_decay)),
                    lr=float(self.args.learning_rate),
                    muon_lr=float(cfg.training.muon_learning_rate),
                    muon_momentum=float(cfg.training.muon_momentum),
                    muon_nesterov=bool(cfg.training.muon_nesterov),
                    muon_ns_steps=int(cfg.training.muon_ns_steps),
                )
            return self.optimizer

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model), padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading train split")
    train = load_split(
        "train",
        splits_dir,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        max_prompt_chars=int(cfg.judge.max_prompt_chars),
        context_truncation=str(cfg.judge.context_truncation),
        include_reasoning=include_reasoning,
        reasoning_max_chars=reasoning_max_chars,
        reasoning_truncation=str(cfg.judge.reasoning_truncation),
        enable_thinking=bool(cfg.judge.enable_thinking),
        limit=None if cfg.train_limit is None else int(cfg.train_limit),
    )
    print(
        f"train rows={len(train.frame)} datasets={len(train.dataset_names)} "
        f"positives={int(train.frame['label'].sum())}"
    )

    print("loading validation split")
    validation = load_split(
        "validation",
        splits_dir,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        max_prompt_chars=int(cfg.judge.max_prompt_chars),
        context_truncation=str(cfg.judge.context_truncation),
        include_reasoning=include_reasoning,
        reasoning_max_chars=reasoning_max_chars,
        reasoning_truncation=str(cfg.judge.reasoning_truncation),
        enable_thinking=bool(cfg.judge.enable_thinking),
        limit=None if cfg.validation_limit is None else int(cfg.validation_limit),
    )
    print(
        f"validation rows={len(validation.frame)} datasets={len(validation.dataset_names)} "
        f"positives={int(validation.frame['label'].sum())}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.model),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.warnings_issued = {}
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    report_to = ["wandb"] if bool(cfg.wandb.enabled) else []
    training_args = GRPOConfig(
        output_dir=str(output_dir / "trainer"),
        run_name=str(cfg.wandb.run_name),
        report_to=report_to,
        num_train_epochs=float(cfg.training.num_train_epochs),
        max_steps=int(cfg.training.max_steps),
        learning_rate=float(cfg.training.learning_rate),
        warmup_ratio=float(cfg.training.warmup_ratio),
        weight_decay=float(cfg.training.weight_decay),
        per_device_train_batch_size=int(cfg.training.per_device_train_batch_size),
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        num_generations=int(cfg.training.num_generations),
        max_prompt_length=int(cfg.training.max_prompt_length),
        max_completion_length=int(cfg.training.max_completion_length),
        bf16=bool(cfg.training.bf16),
        gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
        logging_steps=int(cfg.training.logging_steps),
        logging_first_step=True,
        save_strategy=str(cfg.training.save_strategy),
        save_total_limit=int(cfg.training.save_total_limit),
        beta=float(cfg.training.beta),
        temperature=float(cfg.training.temperature),
        top_p=float(cfg.training.top_p),
        use_vllm=bool(cfg.vllm.enabled),
        vllm_mode=str(cfg.vllm.mode),
        vllm_gpu_memory_utilization=float(cfg.vllm.gpu_memory_utilization),
        vllm_enable_sleep_mode=bool(cfg.vllm.enable_sleep_mode),
        reward_weights=[float(x) for x in cfg.training.reward_weights],
        remove_unused_columns=False,
        seed=int(cfg.seed),
        data_seed=int(cfg.seed),
    )

    metadata: dict[str, Any] = {
        "method": str(cfg.method),
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "versions": {
            package: version_or_missing(package)
            for package in ["trl", "transformers", "peft", "accelerate", "vllm", "wandb"]
        },
        "lora_target_modules": list(cfg.lora.target_modules),
        "prompt": prompt_template,
        "include_reasoning": include_reasoning,
        "reasoning_max_chars": reasoning_max_chars,
        "baseline_threshold": baseline_threshold,
        "train_rows": int(len(train.frame)),
        "validation_rows": int(len(validation.frame)),
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2))

    trainer_cls = MuonGRPOTrainer if str(cfg.training.optimizer) == "muon" else GRPOTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        reward_funcs=[correctness_reward, format_reward],
        train_dataset=package_dataset(train.frame),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("starting GRPO training")
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(output_dir / "adapter")

    print("evaluating validation split")
    predictions, eval_meta = evaluate_model(
        model=trainer.model,
        tokenizer=tokenizer,
        validation=validation,
        batch_size=int(cfg.evaluation.batch_size),
        max_new_tokens=int(cfg.training.max_completion_length),
        rating_min=1,
        rating_max=7,
    )
    threshold, metrics = select_threshold(predictions)
    default_metrics = macro_metrics(predictions, threshold=baseline_threshold)
    predictions_path = output_dir / "validation_predictions.csv"
    generations_path = output_dir / "validation_generations.jsonl"
    write_predictions(predictions_path, predictions, threshold)
    generations_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions.to_dict("records")) + "\n"
    )

    result = {
        **metadata,
        "train_metrics": train_result.metrics,
        "validation": {
            "selected_threshold": threshold,
            "selected_threshold_metrics": metrics,
            "baseline_threshold": baseline_threshold,
            "baseline_threshold_metrics": default_metrics,
            "per_dataset": per_dataset_table(predictions, threshold),
            **eval_meta,
            "predictions_path": predictions_path.as_posix(),
            "generations_path": generations_path.as_posix(),
        },
        "adapter_path": (output_dir / "adapter").as_posix(),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
