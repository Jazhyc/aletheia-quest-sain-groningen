#!/usr/bin/env python3
"""Evaluate one or more privileged-distillation LoRA students with one vLLM."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import (
    append_reference_material,
    build_student_prompt,
    final_assistant_reasoning,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import (
    binary_metrics,
    load_split_config,
    load_labels,
    macro_metrics,
)

PREDICTION_RE = re.compile(r"(?im)\bPrediction\s*:\s*<?([01])>?")
RATING_RE = re.compile(r"(?im)\bRating\s*:\s*<?([1-7])\b>?")
STRICT_RE = re.compile(
    r"(?is)^\s*<reasoning_summary>\s*(.+?)\s*</reasoning_summary>\s*"
    r"Prediction\s*:\s*([01])\s*$"
)
COUNTERFACTUAL_STRICT_RE = re.compile(
    r"(?is)^\s*<reasoning_summary>\s*(.+?)\s*</reasoning_summary>\s*"
    r"<facts>\s*(.+?)\s*</facts>\s*"
    r"<contradiction>\s*(.+?)\s*</contradiction>\s*"
    r"Prediction\s*:\s*([01])\s*$"
)
ATOMIC_AUDIT_STRICT_RE = re.compile(
    r"(?is)^\s*<factual_claims>\s*(?:<claim>\s*"
    r"<quote>.+?</quote>\s*<proposition>.+?</proposition>\s*"
    r"<assessment>\s*(?:true|false|uncertain)\s*</assessment>\s*"
    r"</claim>\s*)*</factual_claims>\s*"
    r"<context_claims>\s*(?:<context_claim>.+?</context_claim>\s*)*"
    r"</context_claims>\s*"
    r"<audit_label_relation>\s*(?:supports|conflicts|insufficient)\s*"
    r"</audit_label_relation>\s*"
    r"<reasoning_summary>\s*.+?</reasoning_summary>\s*"
    r"Prediction\s*:\s*([01])\s*$"
)
PREDICTION_ONLY_STRICT_RE = re.compile(r"(?is)^\s*Prediction\s*:\s*<?([01])>?\s*$")
RATING_STRICT_RE = re.compile(
    r"(?is)^\s*<reasoning_summary>\s*(.+?)\s*</reasoning_summary>\s*"
    r"Rating\s*:\s*<?([1-7])>?\s*$"
)
NATIVE_RATING_STRICT_RE = re.compile(
    r"(?is)^\s*<think>.+?</think>\s*.*"
    r"Rating\s*:\s*<?([1-7])>?\s*$"
)
SUMMARY_RATING_STRICT_RE = re.compile(
    r"(?is)^\s*<reasoning_summary>\s*(.+?)\s*</reasoning_summary>\s*"
    r"Rating\s*:\s*<?([1-7])>?\s*"
    r"Prediction\s*:\s*<?([01])>?\s*$"
)
EMPTY_REASONING_PREFIX = (
    "<reasoning_summary>\n"
    "</reasoning_summary>\n"
    "Prediction:"
)
DIRECT_PREDICTION_PREFIX = "Prediction:"
DIRECT_RATING_PREFIX = "Rating:"
CONTINUOUS_MARGIN_CONDITIONS = ("direct", "empty", "reasoning")
QWEN35_MODEL_IDS = {"Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"}
QWEN35_CANONICAL_PREFIX = "base_model.model.model.language_model.layers."
QWEN35_VISION_EXCLUDE_PATTERN = r".*(visual|vision_tower|merger|patch_embed).*"


def validate_qwen35_adapter_layout(adapter_dir: Path, model_name: str) -> None:
    """Fail before vLLM startup when a Qwen3.5 adapter uses legacy PEFT keys."""
    if model_name not in QWEN35_MODEL_IDS:
        return
    from safetensors import safe_open

    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"{adapter_dir} must contain adapter_config.json and "
            "adapter_model.safetensors"
        )
    adapter_config = json.loads(config_path.read_text())
    if adapter_config.get("exclude_modules") != QWEN35_VISION_EXCLUDE_PATTERN:
        raise ValueError(
            f"{adapter_dir} does not exclude Qwen3.5 visual modules"
        )
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if not keys:
        raise ValueError(f"{weights_path} contains no adapter tensors")
    noncanonical = [
        key for key in keys if not key.startswith(QWEN35_CANONICAL_PREFIX)
    ]
    if noncanonical:
        raise ValueError(
            f"{adapter_dir} has NDIF-incompatible Qwen3.5 LoRA keys; "
            f"first={noncanonical[:3]}"
        )


def parse_prediction(text: str) -> int | None:
    matches = PREDICTION_RE.findall(text)
    return int(matches[-1]) if matches else None


def final_after_thinking(text: str) -> str | None:
    """Return only the answer after a closed native-thinking block."""
    marker = "</think>"
    if marker not in text:
        return None
    return text.rpartition(marker)[2]


def parse_rating(text: str) -> int | None:
    """Return the final explicit 1--7 rating, if present."""
    matches = RATING_RE.findall(text)
    return int(matches[-1]) if matches else None


def rating_to_score(rating: int) -> float:
    """Map an ordinal 1--7 rating onto the evaluator's [0, 1] score range."""
    if not 1 <= rating <= 7:
        raise ValueError(f"rating must be between 1 and 7, got {rating}")
    return (rating - 1) / 6


def rating_prediction_to_score(rating: int, prediction: int) -> float:
    """Map a signed rating to [0, 1], using Prediction only to break rating-4 ties."""
    score = rating_to_score(rating)
    if rating == 4:
        if prediction not in (0, 1):
            raise ValueError(f"prediction must be 0 or 1, got {prediction}")
        return 0.500001 if prediction == 1 else 0.499999
    return score


def prefix_before_prediction(text: str) -> str:
    """Return a completion prefix ending immediately before its final 0/1 token."""
    matches = list(PREDICTION_RE.finditer(text))
    if matches:
        return text[:matches[-1].start(1)]
    return text.rstrip() + "\nPrediction:"


def prefix_before_rating(text: str) -> str:
    """Return a completion prefix ending immediately before its final rating token."""
    matches = list(RATING_RE.finditer(text))
    if matches:
        return text[:matches[-1].start(1)]
    return text.rstrip() + "\nRating:"


def continuous_margin_prompts(
    prompt: str,
    generation: str,
    *,
    direct_prefix: str = DIRECT_PREDICTION_PREFIX,
    empty_reasoning_prefix: str = EMPTY_REASONING_PREFIX,
) -> dict[str, str]:
    """Build matched direct, empty-summary, and post-reasoning scoring prompts."""
    return {
        "direct": prompt + direct_prefix,
        "empty": prompt + empty_reasoning_prefix,
        "reasoning": prompt + prefix_before_prediction(generation),
    }


def continuous_rating_prompts(
    prompt: str,
    generation: str,
) -> dict[str, str]:
    """Build direct, empty-thinking, and post-reasoning rating prefixes."""
    return {
        "direct": prompt + DIRECT_RATING_PREFIX,
        "empty": prompt + "<think>\n</think>\nRating:",
        "reasoning": prompt + prefix_before_rating(generation),
    }


def binary_token_ids(tokenizer: Any) -> list[int]:
    """Return the distinct single-token ids used for literal binary predictions."""
    ids = []
    for text in ("0", "1"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"binary target {text!r} tokenized as {encoded}, expected one token")
        ids.append(int(encoded[0]))
    if len(set(ids)) != 2:
        raise ValueError(f"binary targets must have distinct token ids, got {ids}")
    return ids


def rating_token_ids(tokenizer: Any) -> list[int]:
    """Return distinct single-token ids used for literal ratings 1--7."""
    ids = []
    for text in map(str, range(1, 8)):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"rating target {text!r} tokenized as {encoded}, expected one token"
            )
        ids.append(int(encoded[0]))
    if len(set(ids)) != 7:
        raise ValueError(f"rating targets must have distinct token ids, got {ids}")
    return ids


def logprob_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def binary_score_from_logprobs(
    first_token_logprobs: dict[Any, Any],
    token_ids: list[int],
) -> float | None:
    """Normalize constrained 0/1 token probabilities into P(Prediction=1)."""
    expanded = {
        int(token_id): logprob_value(value)
        for token_id, value in first_token_logprobs.items()
    }
    if any(token_id not in expanded for token_id in token_ids):
        return None
    logit_zero, logit_one = (expanded[token_id] for token_id in token_ids)
    difference = max(-80.0, min(80.0, logit_one - logit_zero))
    return 1.0 / (1.0 + math.exp(-difference))


def rating_score_from_logprobs(
    first_token_logprobs: dict[Any, Any],
    token_ids: list[int],
) -> float | None:
    """Return normalized expected rating, mapped from 1--7 onto [0, 1]."""
    probabilities = rating_probabilities_from_logprobs(
        first_token_logprobs, token_ids
    )
    if probabilities is None:
        return None
    return sum(
        ((rating - 1) / 6) * probability
        for rating, probability in enumerate(probabilities, start=1)
    )


def rating_probabilities_from_logprobs(
    first_token_logprobs: dict[Any, Any],
    token_ids: list[int],
) -> list[float] | None:
    """Normalize constrained 1--7 log probabilities in rating order."""
    expanded = {
        int(token_id): logprob_value(value)
        for token_id, value in first_token_logprobs.items()
    }
    if any(token_id not in expanded for token_id in token_ids):
        return None
    logits = [expanded[token_id] for token_id in token_ids]
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    normalization = sum(weights)
    return [weight / normalization for weight in weights]


def score_binary_prefixes(
    llm: Any,
    prompts: list[str],
    sampling: Any,
    request: Any,
    token_ids: list[int],
) -> tuple[list[float], int, float]:
    """Score constrained binary next-token margins for rendered prefixes."""
    started = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=request)
    elapsed = time.time() - started
    scores = []
    missing = 0
    for output in outputs:
        first_token_logprobs = {}
        if output.outputs and output.outputs[0].logprobs:
            first_token_logprobs = output.outputs[0].logprobs[0] or {}
        score = binary_score_from_logprobs(first_token_logprobs, token_ids)
        if score is None:
            missing += 1
            score = 0.5
        scores.append(score)
    return scores, missing, elapsed


def score_rating_prefixes(
    llm: Any,
    prompts: list[str],
    sampling: Any,
    request: Any,
    token_ids: list[int],
) -> tuple[list[float], list[list[float]], int, float]:
    """Score constrained seven-way expected ratings for rendered prefixes."""
    started = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=request)
    elapsed = time.time() - started
    scores = []
    probability_rows = []
    missing = 0
    for output in outputs:
        first_token_logprobs = {}
        if output.outputs and output.outputs[0].logprobs:
            first_token_logprobs = output.outputs[0].logprobs[0] or {}
        probabilities = rating_probabilities_from_logprobs(
            first_token_logprobs, token_ids
        )
        if probabilities is None:
            missing += 1
            probabilities = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        score = sum(
            ((rating - 1) / 6) * probability
            for rating, probability in enumerate(probabilities, start=1)
        )
        scores.append(score)
        probability_rows.append(probabilities)
    return scores, probability_rows, missing, elapsed


def scenario_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    result = {"all": macro_metrics(frame, 0.5)}
    for scenario in ("instructed", "varied"):
        subset = frame[frame["dataset"].str.contains(f"dev-{scenario}-deception")]
        if not subset.empty:
            result[scenario] = macro_metrics(subset, 0.5)
    return result


def load_retrieval_cache(
    path: Path | None,
    passage_field: str = "passages",
) -> dict[tuple[str, Any], str]:
    if path is None:
        return {}
    references = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        passages = record.get(passage_field) or []
        text = "\n".join(
            f"- {passage.get('title', '')}: {passage.get('text', '')}"
            for passage in passages
            if passage.get("text")
        )
        references[(record["dataset"], record["index"])] = text
    return references


def comparable_student_settings(config: dict[str, Any]) -> tuple[Any, ...]:
    """Normalize legacy configs before checking multi-adapter comparability."""
    student = config["student"]
    return (
        student.get("model"),
        student.get("prompt"),
        student.get("prompt_without_reasoning"),
        student.get("max_prompt_chars"),
        student.get("context_truncation"),
        student.get("include_reasoning", False),
        student.get("reasoning_max_chars", 0),
        student.get("reasoning_truncation", "head_tail"),
        student.get("exclude_final_output_from_context", False),
        student.get("target_format", "summary"),
        student.get("target_mode", "teacher"),
    )


VLLM_MAX_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def vllm_max_lora_rank(configured_ranks: list[int]) -> int:
    """Round adapter rank up to a max_lora_rank accepted by vLLM."""
    if not configured_ranks or min(configured_ranks) < 1:
        raise ValueError("configured LoRA ranks must be positive")
    required = max(configured_ranks)
    for supported in VLLM_MAX_LORA_RANKS:
        if supported >= required:
            return supported
    raise ValueError(f"LoRA rank {required} exceeds vLLM's supported maximum")


def set_reasoning_visibility(config: dict[str, Any], visibility: str) -> None:
    """Apply an inference-only reasoning-visibility ablation in place."""
    if visibility == "configured":
        return
    if visibility == "hidden":
        config["student"]["include_reasoning"] = False
        return
    raise ValueError(f"unsupported reasoning visibility: {visibility}")


def parse_retrieval_condition(
    value: str,
    root: Path,
) -> tuple[str, Path | None, str]:
    """Parse NAME, NAME=PATH, or NAME=PATH#PASSAGE_FIELD conditions."""
    if value == "empty":
        return "empty", None, "passages"
    name, separator, location = value.partition("=")
    if not separator or not name or not location:
        raise ValueError(
            "retrieval conditions must be 'empty' or NAME=PATH[#PASSAGE_FIELD]"
        )
    path_text, field_separator, passage_field = location.rpartition("#")
    if not field_separator:
        path_text, passage_field = location, "passages"
    if not path_text or not passage_field:
        raise ValueError(f"invalid retrieval condition: {value!r}")
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    return name, path.resolve(), passage_field


def parse_thinking_condition(value: str) -> tuple[str, bool, int]:
    """Parse NAME=on|off:MAX_NEW_TOKENS for a shared-session ablation."""
    name, separator, specification = value.partition("=")
    mode, budget_separator, budget_text = specification.partition(":")
    normalized_mode = mode.strip().lower()
    if (
        not separator
        or not name
        or not budget_separator
        or normalized_mode not in {"on", "off", "true", "false"}
    ):
        raise ValueError(
            "thinking conditions must be NAME=on|off:MAX_NEW_TOKENS"
        )
    try:
        max_new_tokens = int(budget_text)
    except ValueError as error:
        raise ValueError("thinking-condition token budget must be an integer") from error
    if max_new_tokens < 1:
        raise ValueError("thinking-condition token budget must be positive")
    return name, normalized_mode in {"on", "true"}, max_new_tokens


def parse_reasoning_input_condition(value: str) -> tuple[str, int, str]:
    """Parse NAME=MAX_CHARS or NAME=MAX_CHARS:TRUNCATION."""
    name, separator, settings = value.partition("=")
    if not separator or not name or not settings:
        raise ValueError("reasoning input conditions must be NAME=MAX_CHARS[:TRUNCATION]")
    max_chars_text, mode_separator, mode = settings.partition(":")
    try:
        max_chars = int(max_chars_text)
    except ValueError as error:
        raise ValueError(f"invalid reasoning max chars: {max_chars_text!r}") from error
    if max_chars <= 0:
        raise ValueError("reasoning max chars must be positive")
    mode = mode if mode_separator else "head_tail"
    if mode not in {
        "head",
        "tail",
        "head_tail",
        "head_tail_25",
        "head_tail_75",
    }:
        raise ValueError(f"invalid reasoning truncation: {mode!r}")
    return name, max_chars, mode


def parse_prompt_condition(value: str, root: Path) -> tuple[str, Path]:
    """Parse NAME=CONFIG_PATH for a shared-session prompt sweep."""
    name, separator, path_text = value.partition("=")
    if not separator or not name or not path_text:
        raise ValueError("prompt conditions must be NAME=CONFIG_PATH")
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    return name, path.resolve()


def apply_student_prompt_config(config: dict[str, Any], path: Path) -> None:
    """Override only inference prompt rendering fields from a Hydra config."""
    prompt_config = yaml.safe_load(path.read_text())["student"]
    for field in (
        "prompt",
        "prompt_without_reasoning",
        "include_reasoning",
        "reasoning_max_chars",
        "reasoning_truncation",
        "exclude_final_output_from_context",
        "target_mode",
        "target_format",
    ):
        if field in prompt_config:
            config["student"][field] = prompt_config[field]


def strict_pattern_for_config(config: dict[str, Any]) -> re.Pattern[str]:
    """Select the exact-output validator for one inference prompt config."""
    student = config["student"]
    if student.get("target_mode") == "prediction_only":
        return PREDICTION_ONLY_STRICT_RE
    if student.get("target_format") == "counterfactual":
        return COUNTERFACTUAL_STRICT_RE
    if student.get("target_format") == "atomic_audit":
        return ATOMIC_AUDIT_STRICT_RE
    if student.get("target_format") == "summary_rating":
        return SUMMARY_RATING_STRICT_RE
    if student.get("target_format") == "native_rating":
        return NATIVE_RATING_STRICT_RE
    if student.get("target_format") == "rating":
        return RATING_STRICT_RE
    return STRICT_RE


def output_mode_for_config(config: dict[str, Any]) -> str:
    """Select binary or ordinal parsing from the inference prompt contract."""
    target_format = config["student"].get("target_format")
    if target_format in {"rating", "native_rating"}:
        return "rating"
    if target_format == "summary_rating":
        return "rating_prediction"
    return "binary"


def load_records(
    split: str,
    splits_dir: Path,
    config: dict[str, Any],
    tokenizer: Any,
    references: dict[tuple[str, Any], str] | None = None,
    *,
    append_empty_reference: bool = False,
    enable_thinking: bool = False,
) -> pd.DataFrame:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    # Checked-in split manifests use project-relative labels such as
    # dev_splits/labels/validation/*.csv. Resolve them from the parent of the
    # selected splits directory, not from this script's checkout/worktree.
    for dataset_cfg in load_split_config(
        splits_dir / f"dry.{split}.yaml",
        splits_dir.parent,
    ):
        labels = load_labels(dataset_cfg)
        label_by_index = dict(zip(labels["index"], labels["label"], strict=True))
        dataset = load_dataset(dataset_cfg.name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        found: set[Any] = set()
        for row in dataset:
            index = row["index"]
            if index not in label_by_index:
                continue
            reasoning_present = bool(final_assistant_reasoning(row["messages"]).strip())
            prompt_template = config["student"]["prompt"]
            if not reasoning_present and config["student"].get("prompt_without_reasoning"):
                prompt_template = config["student"]["prompt_without_reasoning"]
            raw_prompt = build_student_prompt(
                row["messages"],
                prompt_template,
                int(config["student"]["max_prompt_chars"]),
                config["student"]["context_truncation"],
                include_reasoning=bool(
                    config["student"].get("include_reasoning", False)
                ),
                reasoning_max_chars=int(
                    config["student"].get("reasoning_max_chars", 0)
                ),
                reasoning_truncation=str(
                    config["student"].get("reasoning_truncation", "head_tail")
                ),
                exclude_final_output_from_context=bool(
                    config["student"].get(
                        "exclude_final_output_from_context", False
                    )
                ),
            )
            reference = (references or {}).get((dataset_cfg.name, index), "")
            if reference or append_empty_reference:
                raw_prompt = append_reference_material(raw_prompt, reference)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            rows.append({
                "dataset": dataset_cfg.name,
                "index": index,
                "label": int(label_by_index[index]),
                "reasoning_present": reasoning_present,
                "prompt": prompt,
            })
            found.add(index)
        if len(found) != len(labels):
            raise RuntimeError(f"{dataset_cfg.name}: found {len(found)}/{len(labels)} labeled rows")
    return pd.DataFrame(rows)


def lora_request_or_none(
    adapter_dir: Path,
    lora_id: int,
    *,
    use_lora: bool,
) -> Any | None:
    """Construct a vLLM LoRA request unless this is a matched base control."""
    if not use_lora:
        return None
    from vllm.lora.request import LoRARequest

    return LoRARequest(adapter_dir.parent.name, lora_id, adapter_dir.as_posix())


def summarize_lora_effect(
    adapter: pd.DataFrame,
    base: pd.DataFrame,
    score_column: str,
) -> dict[str, float | int | str]:
    """Fingerprint matched scores and reject a silently ignored LoRA."""
    keys = ["dataset", "index"]
    if not adapter[keys].equals(base[keys]):
        raise ValueError("base and adapter fingerprint rows are not aligned")
    adapter_scores = adapter[score_column].to_numpy(dtype=np.float32)
    base_scores = base[score_column].to_numpy(dtype=np.float32)
    differences = np.abs(adapter_scores - base_scores)
    if np.all(differences <= 1.0e-7):
        raise RuntimeError(
            "LoRA effect fingerprint failed: adapter and base scores are "
            f"identical within 1e-7 for {len(differences)} rows"
        )
    return {
        "score_column": score_column,
        "rows": len(differences),
        "exact_equal": int(np.equal(adapter_scores, base_scores).sum()),
        "mean_absolute_difference": float(differences.mean()),
        "max_absolute_difference": float(differences.max()),
        "adapter_sha256_float32": hashlib.sha256(
            adapter_scores.tobytes()
        ).hexdigest(),
        "base_sha256_float32": hashlib.sha256(base_scores.tobytes()).hexdigest(),
    }


def evaluate_adapter(
    llm: Any,
    sampling: Any,
    records: pd.DataFrame,
    adapter_dir: Path,
    lora_id: int,
    strict_re: re.Pattern[str] = STRICT_RE,
    output_mode: str = "binary",
    *,
    margin_sampling: Any | None = None,
    binary_ids: list[int] | None = None,
    rating_ids: list[int] | None = None,
    margin_conditions: tuple[str, ...] = CONTINUOUS_MARGIN_CONDITIONS,
    direct_prefix: str = DIRECT_PREDICTION_PREFIX,
    empty_reasoning_prefix: str = EMPTY_REASONING_PREFIX,
    require_closed_thinking: bool = False,
    use_lora: bool = True,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    request = lora_request_or_none(adapter_dir, lora_id, use_lora=use_lora)
    started = time.time()
    outputs = llm.generate(records["prompt"].tolist(), sampling, lora_request=request)
    elapsed = time.time() - started
    evaluated = records.drop(columns="prompt").copy()
    evaluated["prompt_sha256"] = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for prompt in records["prompt"]
    ]
    generations = [output.outputs[0].text if output.outputs else "" for output in outputs]
    evaluated["finish_reason"] = [
        None
        if not output.outputs
        else str(output.outputs[0].finish_reason)
        for output in outputs
    ]
    evaluated["generation_tokens"] = [
        0 if not output.outputs else len(output.outputs[0].token_ids)
        for output in outputs
    ]
    parsed_generations = generations
    if require_closed_thinking:
        parsed_generations = [final_after_thinking(text) or "" for text in generations]
        evaluated["final_generation"] = parsed_generations
    if output_mode == "rating":
        ratings = [parse_rating(text) for text in parsed_generations]
        predictions = [None if value is None else int(value >= 4) for value in ratings]
        scores = [0.0 if value is None else rating_to_score(value) for value in ratings]
        evaluated["rating"] = ratings
    elif output_mode == "rating_prediction":
        ratings = [parse_rating(text) for text in parsed_generations]
        predictions = [parse_prediction(text) for text in parsed_generations]
        scores = [
            0.0
            if rating is None or prediction is None
            else rating_prediction_to_score(rating, prediction)
            for rating, prediction in zip(ratings, predictions, strict=True)
        ]
        evaluated["rating"] = ratings
        evaluated["prediction_score"] = [
            float(value) if value is not None else 0.0 for value in predictions
        ]
        evaluated["rating_prediction_conflict"] = [
            False
            if rating is None or prediction is None or rating == 4
            else (rating >= 5) != bool(prediction)
            for rating, prediction in zip(ratings, predictions, strict=True)
        ]
    elif output_mode == "binary":
        predictions = [parse_prediction(text) for text in parsed_generations]
        scores = [float(value) if value is not None else 0.0 for value in predictions]
    else:
        raise ValueError(f"unsupported output mode: {output_mode!r}")
    evaluated["prediction"] = predictions
    evaluated["score"] = scores
    if output_mode == "rating_prediction":
        evaluated["parse_error"] = [
            rating is None or prediction is None
            for rating, prediction in zip(ratings, predictions, strict=True)
        ]
    else:
        evaluated["parse_error"] = [value is None for value in predictions]
    evaluated["format_valid"] = [
        strict_re.fullmatch(text) is not None for text in parsed_generations
    ]
    evaluated["generation"] = generations
    timing: dict[str, float | int] = {"generation_seconds": elapsed}

    if margin_sampling is not None:
        if (binary_ids is None) == (rating_ids is None):
            raise ValueError(
                "exactly one of binary_ids or rating_ids is required for margins"
            )
        source_prompts = records["prompt"].tolist()
        if rating_ids is not None:
            margin_prompts = [
                continuous_rating_prompts(prompt, generation)
                for prompt, generation in zip(
                    source_prompts, generations, strict=True
                )
            ]
            score_prefixes = score_rating_prefixes
            target_ids = rating_ids
        else:
            margin_prompts = [
                continuous_margin_prompts(
                    prompt,
                    generation,
                    direct_prefix=direct_prefix,
                    empty_reasoning_prefix=empty_reasoning_prefix,
                )
                for prompt, generation in zip(
                    source_prompts, generations, strict=True
                )
            ]
            score_prefixes = score_binary_prefixes
            target_ids = binary_ids
        unknown = set(margin_conditions).difference(CONTINUOUS_MARGIN_CONDITIONS)
        if unknown:
            raise ValueError(f"unknown continuous margin conditions: {sorted(unknown)}")
        for condition in margin_conditions:
            condition_prompts = [prompts[condition] for prompts in margin_prompts]
            if rating_ids is not None:
                (
                    scores,
                    probability_rows,
                    missing,
                    condition_elapsed,
                ) = score_rating_prefixes(
                    llm,
                    condition_prompts,
                    margin_sampling,
                    request,
                    target_ids,
                )
                for rating in range(1, 8):
                    evaluated[
                        f"{condition}_rating_{rating}_prob"
                    ] = [
                        probabilities[rating - 1]
                        for probabilities in probability_rows
                    ]
                evaluated[f"{condition}_p7_margin_score"] = [
                    probabilities[6] for probabilities in probability_rows
                ]
            else:
                scores, missing, condition_elapsed = score_prefixes(
                    llm, condition_prompts, margin_sampling, request, target_ids
                )
            evaluated[f"{condition}_margin_score"] = scores
            timing[f"{condition}_margin_seconds"] = condition_elapsed
            timing[f"{condition}_margin_missing"] = missing
    return evaluated, timing


def metrics_for_score(evaluated: pd.DataFrame, score_column: str) -> dict[str, Any]:
    scored = evaluated[["dataset", "index", "label", score_column]].rename(
        columns={score_column: "score"}
    )
    return scenario_metrics(scored)


def max_aggregate_evaluations(
    evaluations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Align prompt conditions and max their binary deception scores."""
    if len(evaluations) < 2:
        raise ValueError("max aggregation requires at least two conditions")
    names = list(evaluations)
    first = evaluations[names[0]].copy()
    keys = ["dataset", "index"]
    if first.duplicated(keys).any():
        raise ValueError(f"duplicate row keys in condition {names[0]!r}")
    base = first.set_index(keys, drop=False)
    result = base[["dataset", "index", "label"]].copy()
    score_columns = []
    parse_columns = []
    format_columns = []
    generation_columns = []
    prompt_hash_columns = []
    margin_suffixes = sorted(
        column for column in first.columns if column.endswith("_margin_score")
    )
    margin_columns_by_suffix = {suffix: [] for suffix in margin_suffixes}
    for name, frame in evaluations.items():
        if frame.duplicated(keys).any():
            raise ValueError(f"duplicate row keys in condition {name!r}")
        aligned = frame.set_index(keys, drop=False).reindex(base.index)
        if aligned["label"].isna().any() or not aligned["label"].equals(base["label"]):
            raise ValueError(f"row keys or labels differ for condition {name!r}")
        score_column = f"{name}_score"
        parse_column = f"{name}_parse_error"
        format_column = f"{name}_format_valid"
        generation_column = f"{name}_generation"
        prompt_hash_column = f"{name}_prompt_sha256"
        result[score_column] = aligned["score"].astype(float)
        result[parse_column] = aligned["parse_error"].astype(bool)
        result[format_column] = aligned["format_valid"].astype(bool)
        result[generation_column] = aligned["generation"]
        result[prompt_hash_column] = aligned["prompt_sha256"]
        score_columns.append(score_column)
        parse_columns.append(parse_column)
        format_columns.append(format_column)
        generation_columns.append(generation_column)
        prompt_hash_columns.append(prompt_hash_column)
        for suffix in margin_suffixes:
            if suffix not in aligned:
                raise ValueError(
                    f"condition {name!r} is missing margin column {suffix!r}"
                )
            column = f"{name}_{suffix}"
            result[column] = aligned[suffix].astype(float)
            margin_columns_by_suffix[suffix].append(column)
    result["score"] = result[score_columns].max(axis=1)
    for suffix, columns in margin_columns_by_suffix.items():
        result[suffix] = result[columns].max(axis=1)
    result["prediction"] = (result["score"] >= 0.5).astype(float)
    result["parse_error"] = result[parse_columns].all(axis=1)
    result["format_valid"] = result[format_columns].any(axis=1)
    result["generation"] = result[generation_columns].apply(
        lambda row: json.dumps(dict(zip(names, row, strict=True))), axis=1
    )
    result["prompt_sha256"] = result[prompt_hash_columns].apply(
        lambda row: json.dumps(dict(zip(names, row, strict=True))), axis=1
    )
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", action="append", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--retrieval-cache", type=Path)
    parser.add_argument(
        "--retrieval-condition",
        action="append",
        default=[],
        help="Repeat empty or NAME=PATH[#PASSAGE_FIELD] for a shared-session sweep.",
    )
    parser.add_argument(
        "--reasoning-input-condition",
        action="append",
        default=[],
        help="repeat NAME=MAX_CHARS[:head|tail|head_tail] for a shared prompt sweep",
    )
    parser.add_argument(
        "--prompt-condition",
        action="append",
        default=[],
        help="repeat NAME=CONFIG_PATH for a shared-session prompt sweep",
    )
    parser.add_argument(
        "--thinking-condition",
        action="append",
        default=[],
        help="repeat NAME=on|off:MAX_NEW_TOKENS for a shared-session ablation",
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--aggregate-max",
        action="store_true",
        help="write a max-score ensemble across all prompt/input conditions",
    )
    parser.add_argument(
        "--reasoning-visibility",
        default="configured",
        choices=["configured", "hidden"],
        help="optionally hide configured assistant reasoning at inference time",
    )
    parser.add_argument(
        "--continuous-margins",
        action="store_true",
        help=(
            "score constrained 0/1 logits with direct, empty-summary, and "
            "generated-reasoning prefixes"
        ),
    )
    parser.add_argument(
        "--continuous-rating-margins",
        action="store_true",
        help=(
            "score constrained 1--7 logits as normalized expected ratings with "
            "direct, empty-thinking, and generated-reasoning prefixes"
        ),
    )
    parser.add_argument(
        "--continuous-margin-condition",
        action="append",
        choices=CONTINUOUS_MARGIN_CONDITIONS,
        default=[],
        help=(
            "repeat to score only selected margin conditions; defaults to all "
            "three when --continuous-margins is enabled"
        ),
    )
    parser.add_argument(
        "--base-model-control",
        action="store_true",
        help="use the saved adapter prompt/tokenizer without applying its LoRA weights",
    )
    parser.add_argument(
        "--verify-lora-effect",
        action="store_true",
        help=(
            "also score the matched base model in the same vLLM session and "
            "fail if its float32 score fingerprint matches the adapter"
        ),
    )
    parser.add_argument(
        "--prompt-without-reasoning-config",
        type=Path,
        help="use student.prompt from this config only when a row has no trace",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        help="override the saved primary student prompt settings from this config",
    )
    args = parser.parse_args()
    if args.retrieval_cache is not None and args.retrieval_condition:
        parser.error("use either --retrieval-cache or --retrieval-condition, not both")
    if args.continuous_margins and args.continuous_rating_margins:
        parser.error(
            "--continuous-margins and --continuous-rating-margins are exclusive"
        )
    if args.continuous_margin_condition and not (
        args.continuous_margins or args.continuous_rating_margins
    ):
        parser.error(
            "--continuous-margin-condition requires a continuous-margin mode"
        )
    if args.verify_lora_effect and args.base_model_control:
        parser.error("--verify-lora-effect cannot be combined with --base-model-control")
    margin_conditions = tuple(
        dict.fromkeys(
            args.continuous_margin_condition or CONTINUOUS_MARGIN_CONDITIONS
        )
    )
    condition_families = sum(bool(values) for values in (
        args.retrieval_condition,
        args.reasoning_input_condition,
        args.prompt_condition,
        args.thinking_condition,
    ))
    if condition_families > 1 or (
        args.retrieval_cache is not None
        and (
            args.reasoning_input_condition
            or args.prompt_condition
            or args.thinking_condition
        )
    ):
        parser.error("retrieval, reasoning-input, and prompt conditions cannot be crossed")
    if args.aggregate_max and not (
        args.retrieval_condition
        or args.reasoning_input_condition
        or args.prompt_condition
        or args.thinking_condition
    ):
        parser.error("--aggregate-max requires at least two named conditions")

    adapter_dirs = [path.resolve() for path in args.adapter_dir]
    configs = [yaml.safe_load((path.parent / "config.yaml").read_text()) for path in adapter_dirs]
    if not args.base_model_control:
        for adapter_dir, config in zip(adapter_dirs, configs, strict=True):
            validate_qwen35_adapter_layout(
                adapter_dir,
                str(config["student"]["model"]),
            )
    if args.prompt_config is not None:
        for config in configs:
            apply_student_prompt_config(config, args.prompt_config.resolve())
    if args.prompt_without_reasoning_config is not None:
        fallback_config = yaml.safe_load(
            args.prompt_without_reasoning_config.resolve().read_text()
        )
        fallback_prompt = fallback_config["student"]["prompt"]
        for config in configs:
            config["student"]["prompt_without_reasoning"] = fallback_prompt
    for config in configs:
        set_reasoning_visibility(config, args.reasoning_visibility)
    first = configs[0]
    for config in configs[1:]:
        if comparable_student_settings(config) != comparable_student_settings(first):
            raise SystemExit("student prompt/model settings differ across adapters")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(adapter_dirs[0])
    multi_condition = bool(
        args.retrieval_condition
        or args.reasoning_input_condition
        or args.prompt_condition
        or args.thinking_condition
    )
    if args.retrieval_condition:
        condition_specs = [
            (*parse_retrieval_condition(value, ROOT), None, None, None, None, None)
            for value in args.retrieval_condition
        ]
    elif args.reasoning_input_condition:
        condition_specs = [
            (name, None, "passages", max_chars, mode, None, None, None)
            for name, max_chars, mode in (
                parse_reasoning_input_condition(value)
                for value in args.reasoning_input_condition
            )
        ]
    elif args.prompt_condition:
        condition_specs = [
            (name, None, "passages", None, None, path, None, None)
            for name, path in (
                parse_prompt_condition(value, ROOT)
                for value in args.prompt_condition
            )
        ]
    elif args.thinking_condition:
        condition_specs = [
            (name, None, "passages", None, None, None, enabled, max_tokens)
            for name, enabled, max_tokens in (
                parse_thinking_condition(value)
                for value in args.thinking_condition
            )
        ]
    else:
        condition_specs = [(
            "default",
            args.retrieval_cache.resolve() if args.retrieval_cache is not None else None,
            "passages",
            None,
            None,
            None,
            None,
            None,
        )]
    names = [name for name, *_ in condition_specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate condition names: {names}")
    records_by_condition = {}
    strict_re_by_condition = {}
    output_mode_by_condition = {}
    for (
        condition_name,
        retrieval_path,
        passage_field,
        reasoning_max_chars,
        reasoning_mode,
        prompt_path,
        enable_thinking,
        condition_max_tokens,
    ) in condition_specs:
        references = load_retrieval_cache(retrieval_path, passage_field=passage_field)
        condition_config = copy.deepcopy(first)
        if reasoning_max_chars is not None:
            condition_config["student"]["include_reasoning"] = True
            condition_config["student"]["reasoning_max_chars"] = reasoning_max_chars
            condition_config["student"]["reasoning_truncation"] = reasoning_mode
        if prompt_path is not None:
            apply_student_prompt_config(condition_config, prompt_path)
        records = load_records(
            args.split,
            args.splits_dir.resolve(),
            condition_config,
            tokenizer,
            references=references,
            append_empty_reference=bool(args.retrieval_condition),
            enable_thinking=bool(enable_thinking),
        )
        records_by_condition[condition_name] = records
        strict_re_by_condition[condition_name] = strict_pattern_for_config(
            condition_config
        )
        output_mode_by_condition[condition_name] = output_mode_for_config(
            condition_config
        )
        print(
            f"loaded condition={condition_name} {len(records)} rows across "
            f"{records['dataset'].nunique()} datasets "
            f"thinking={bool(enable_thinking)} "
            f"max_new_tokens={condition_max_tokens or args.max_new_tokens}",
            flush=True,
        )
    llm = LLM(
        model=first["student"]["model"],
        tokenizer=adapter_dirs[0].as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=vllm_max_lora_rank([
            int(config["student"]["lora"]["r"]) for config in configs
        ]),
        max_model_len=args.max_model_len,
    )
    sampling_by_condition = {
        name: SamplingParams(
            max_tokens=condition_max_tokens or args.max_new_tokens,
            temperature=0.0,
        )
        for name, *_, condition_max_tokens in condition_specs
    }
    binary_ids = None
    rating_ids = None
    margin_sampling = None
    if args.continuous_margins or args.continuous_rating_margins:
        target_ids = (
            binary_token_ids(tokenizer)
            if args.continuous_margins
            else rating_token_ids(tokenizer)
        )
        if args.continuous_margins:
            binary_ids = target_ids
        else:
            rating_ids = target_ids
        margin_sampling = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=len(target_ids),
            logprob_token_ids=target_ids,
            allowed_token_ids=target_ids,
        )
    for lora_id, (adapter_dir, config) in enumerate(zip(adapter_dirs, configs, strict=True), 1):
        evaluated_by_condition = {}
        timing_by_condition = {}
        for (
            condition_name,
            retrieval_path,
            passage_field,
            reasoning_max_chars,
            reasoning_mode,
            prompt_path,
            enable_thinking,
            condition_max_tokens,
        ) in condition_specs:
            records = records_by_condition[condition_name]
            evaluated, timing = evaluate_adapter(
                llm,
                sampling_by_condition[condition_name],
                records,
                adapter_dir,
                lora_id,
                strict_re=strict_re_by_condition[condition_name],
                output_mode=output_mode_by_condition[condition_name],
                margin_sampling=margin_sampling,
                binary_ids=binary_ids,
                rating_ids=rating_ids,
                margin_conditions=margin_conditions,
                require_closed_thinking=bool(enable_thinking),
                use_lora=not args.base_model_control,
            )
            lora_effect = None
            if args.verify_lora_effect:
                base_evaluated, base_timing = evaluate_adapter(
                    llm,
                    sampling_by_condition[condition_name],
                    records,
                    adapter_dir,
                    lora_id,
                    strict_re=strict_re_by_condition[condition_name],
                    output_mode=output_mode_by_condition[condition_name],
                    margin_sampling=margin_sampling,
                    binary_ids=binary_ids,
                    rating_ids=rating_ids,
                    margin_conditions=margin_conditions,
                    require_closed_thinking=bool(enable_thinking),
                    use_lora=False,
                )
                effect_score_column = (
                    "direct_margin_score"
                    if "direct_margin_score" in evaluated
                    else "score"
                )
                lora_effect = summarize_lora_effect(
                    evaluated,
                    base_evaluated,
                    effect_score_column,
                )
                lora_effect["base_score_seconds"] = float(sum(
                    value
                    for key, value in base_timing.items()
                    if key.endswith("_seconds")
                ))
                print(f"lora_effect={lora_effect}", flush=True)
            evaluated_by_condition[condition_name] = evaluated
            timing_by_condition[condition_name] = timing
            method = config["method"]
            output_dir = adapter_dir.parent / (args.run_name or args.split)
            if multi_condition:
                output_dir = output_dir / condition_name
            output_dir.mkdir(parents=True, exist_ok=True)
            evaluated.to_json(output_dir / "generations.jsonl", orient="records", lines=True)
            per_dataset = {}
            for dataset, group in evaluated.groupby("dataset", sort=True):
                per_dataset[dataset] = binary_metrics(
                    group["label"].to_numpy(), group["score"].to_numpy(), 0.5
                )
            if output_mode_by_condition[condition_name] == "rating_prediction":
                score_metrics = {
                    "generated_rating": metrics_for_score(evaluated, "score"),
                    "generated_prediction": metrics_for_score(
                        evaluated, "prediction_score"
                    ),
                }
                primary_score_name = "generated_rating"
            elif output_mode_by_condition[condition_name] == "rating":
                score_metrics = {
                    "generated_rating": metrics_for_score(evaluated, "score")
                }
                primary_score_name = "generated_rating"
            else:
                score_metrics = {
                    "generated_binary": metrics_for_score(evaluated, "score")
                }
                primary_score_name = "generated_binary"
            if args.continuous_margins or args.continuous_rating_margins:
                score_metrics.update({
                    f"{condition}_margin": metrics_for_score(
                        evaluated, f"{condition}_margin_score"
                    )
                    for condition in margin_conditions
                })
            if args.continuous_rating_margins:
                score_metrics.update({
                    f"{condition}_p7_margin": metrics_for_score(
                        evaluated, f"{condition}_p7_margin_score"
                    )
                    for condition in margin_conditions
                })
            total_elapsed = float(sum(
                value for key, value in timing.items() if key.endswith("_seconds")
            ))
            result = {
                "method": method,
                "split": args.split,
                "input_condition": condition_name,
                "retrieval_condition": (
                    condition_name if args.retrieval_condition else "default"
                ),
                "retrieval_passage_field": passage_field,
                "reasoning_input_condition": (
                    condition_name if args.reasoning_input_condition else None
                ),
                "prompt_condition": (
                    condition_name if args.prompt_condition else None
                ),
                "enable_thinking": bool(enable_thinking),
                "output_mode": output_mode_by_condition[condition_name],
                "reasoning_max_chars": reasoning_max_chars,
                "reasoning_truncation": reasoning_mode,
                "learning_rate": config["student"]["training"]["learning_rate"],
                "metrics": score_metrics[primary_score_name],
                "score_metrics": score_metrics,
                "per_dataset": per_dataset,
                "parse_errors": int(evaluated["parse_error"].sum()),
                "format_valid": int(evaluated["format_valid"].sum()),
                "length_capped": int(
                    evaluated["finish_reason"].eq("length").sum()
                ),
                "generation_tokens": {
                    "mean": float(evaluated["generation_tokens"].mean()),
                    "p50": float(evaluated["generation_tokens"].quantile(0.5)),
                    "p95": float(evaluated["generation_tokens"].quantile(0.95)),
                    "max": int(evaluated["generation_tokens"].max()),
                },
                "rating_prediction_conflicts": int(
                    evaluated.get(
                        "rating_prediction_conflict",
                        pd.Series(False, index=evaluated.index),
                    ).sum()
                ),
                "rows": len(evaluated),
                "score_seconds": total_elapsed,
                "rows_per_second": len(evaluated) / total_elapsed,
                "timing": timing,
                "max_new_tokens": condition_max_tokens or args.max_new_tokens,
                "retrieval_cache": retrieval_path.as_posix() if retrieval_path else None,
                "reasoning_visibility": args.reasoning_visibility,
                "base_model_control": args.base_model_control,
                "lora_effect": lora_effect,
                "prompt_without_reasoning_config": (
                    args.prompt_without_reasoning_config.resolve().as_posix()
                    if args.prompt_without_reasoning_config is not None
                    else None
                ),
                "prompt_config": (
                    args.prompt_config.resolve().as_posix()
                    if args.prompt_config is not None
                    else None
                ),
            }
            (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            metrics = result["metrics"]
            print(
                f"{method} condition={condition_name} lr={result['learning_rate']} "
                f"all={metrics['all']} instructed={metrics.get('instructed')} "
                f"varied={metrics.get('varied')} parse_errors={result['parse_errors']} "
                f"time={total_elapsed:.1f}s",
                flush=True,
            )
            if args.continuous_margins or args.continuous_rating_margins:
                print(f"continuous_margin_metrics={score_metrics}", flush=True)
        if args.aggregate_max:
            aggregated = max_aggregate_evaluations(evaluated_by_condition)
            output_dir = adapter_dir.parent / (args.run_name or args.split) / "max_aggregate"
            output_dir.mkdir(parents=True, exist_ok=True)
            aggregated.to_json(
                output_dir / "generations.jsonl", orient="records", lines=True
            )
            per_dataset = {
                dataset: binary_metrics(
                    group["label"].to_numpy(), group["score"].to_numpy(), 0.5
                )
                for dataset, group in aggregated.groupby("dataset", sort=True)
            }
            total_elapsed = float(sum(
                value
                for timing in timing_by_condition.values()
                for key, value in timing.items()
                if key.endswith("_seconds")
            ))
            aggregate_margin_suffixes = sorted(
                column
                for column in aggregated.columns
                if column.endswith("_margin_score")
                and not any(
                    column.startswith(f"{name}_")
                    for name in evaluated_by_condition
                )
            )
            result = {
                "method": config["method"],
                "split": args.split,
                "input_condition": "max_aggregate",
                "member_conditions": list(evaluated_by_condition),
                "metrics": metrics_for_score(aggregated, "score"),
                "score_metrics": {
                    "generated_rating": metrics_for_score(aggregated, "score"),
                    **{
                        suffix.removesuffix("_score"): metrics_for_score(
                            aggregated, suffix
                        )
                        for suffix in aggregate_margin_suffixes
                    },
                },
                "per_dataset": per_dataset,
                "parse_errors": int(aggregated["parse_error"].sum()),
                "format_valid": int(aggregated["format_valid"].sum()),
                "rows": len(aggregated),
                "score_seconds": total_elapsed,
                "rows_per_second": len(aggregated) / total_elapsed,
                "timing_by_condition": timing_by_condition,
                "max_new_tokens": args.max_new_tokens,
            }
            (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            metrics = result["metrics"]
            print(
                f"{config['method']} condition=max_aggregate "
                f"all={metrics['all']} instructed={metrics.get('instructed')} "
                f"varied={metrics.get('varied')} parse_errors={result['parse_errors']} "
                f"time={total_elapsed:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
