#!/usr/bin/env python3
"""Run the frozen Q397 reasoning-prompt sweep through optimized NDIF sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_adapter_reasoning_ndif import run as reasoning_ndif  # noqa: E402
from experiments.phoenix_adapter_validation_ndif.run import (  # noqa: E402
    MODEL_ID,
    binary_token_ids,
    hosted_metadata,
    load_credentials,
)
from experiments.q397_reasoning_router.analyze import (  # noqa: E402
    comparison,
    logit,
    sigmoid,
    summarize,
)
from experiments.q397_reasoning_router.run_ndif import (  # noqa: E402
    ADAPTER,
    load_split_records,
)
from submission.phoenix_wright import (  # noqa: E402
    has_binary_prediction,
    prefix_before_prediction,
)


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
REASONING_PROMPTS = ("summary_baseline", "claim_check", "balanced_audit")
MAX_NEW_TOKENS = 192
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/blackbox/q397_reasoning_prompt_sweep_ndif_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generation-batches-per-session", type=int, default=32)
    parser.add_argument("--score-batches-per-session", type=int, default=0)
    return parser.parse_args()


def prompt_template(name: str) -> str:
    config = yaml.safe_load((PROMPT_DIR / f"{name}.yaml").read_text())
    return str(config["student"]["prompt"])


def split_replies(
    replies: list[str],
    rows: int,
) -> dict[str, list[str]]:
    expected = rows * len(REASONING_PROMPTS)
    if len(replies) != expected:
        raise ValueError(f"expected {expected} replies, found {len(replies)}")
    return {
        name: replies[offset * rows : (offset + 1) * rows]
        for offset, name in enumerate(REASONING_PROMPTS)
    }


def reply_audit(tokenizer: Any, replies: list[str]) -> dict[str, Any]:
    lengths = [
        len(tokenizer.encode(reply, add_special_tokens=False))
        for reply in replies
    ]
    return {
        "fallback_boundaries": int(
            sum(not has_binary_prediction(reply) for reply in replies)
        ),
        "reply_tokens": {
            "minimum": min(lengths),
            "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)),
            "maximum": max(lengths),
        },
    }


def main() -> None:
    args = parse_args()
    load_credentials()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_split_records(
        args.splits_dir.resolve(),
        "validation",
        args.limit,
    )

    from nnsight import LanguageModel

    metadata = hosted_metadata(ADAPTER.repo_id)
    print(
        f"starting adapter={ADAPTER.name} revision={metadata['revision']}",
        flush=True,
    )
    model = LanguageModel(MODEL_ID, peft=ADAPTER.repo_id)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = binary_token_ids(tokenizer)

    generation_prompts = {
        name: reasoning_ndif.render_generation_prompts(
            records,
            tokenizer,
            prompt_template(name),
        )
        for name in REASONING_PROMPTS
    }
    flat_prompts = [
        prompt
        for name in REASONING_PROMPTS
        for prompt in generation_prompts[name]
    ]
    reasoning_ndif.MAX_NEW_TOKENS = MAX_NEW_TOKENS
    flat_replies, generation_audit = reasoning_ndif.generate_reasoning(
        model,
        tokenizer,
        flat_prompts,
        batches_per_session=args.generation_batches_per_session,
    )
    replies = split_replies(flat_replies, len(records))

    binary_prompts = reasoning_ndif.render_generation_prompts(
        records,
        tokenizer,
        prompt_template("binary"),
    )
    prompt_sets = {
        "direct": [prompt + "Prediction:" for prompt in binary_prompts],
        **{
            name: [
                prompt + prefix_before_prediction(reply)
                for prompt, reply in zip(
                    generation_prompts[name],
                    replies[name],
                    strict=True,
                )
            ]
            for name in REASONING_PROMPTS
        },
    }
    scores, score_audit = reasoning_ndif.score_prompt_sets(
        model,
        tokenizer,
        label_ids,
        prompt_sets,
        batches_per_session=args.score_batches_per_session,
    )

    direct = scores["direct"]
    report: dict[str, Any] = {
        "method": "q397_reasoning_prompt_sweep_ndif_v1",
        "split": "validation",
        "rows": len(records),
        "datasets": int(records["dataset"].nunique()),
        "adapter": {
            "name": ADAPTER.name,
            "repo_id": ADAPTER.repo_id,
            "hosted": metadata,
        },
        "protocol": {
            "direct_weight": 0.6,
            "reasoning_weight": 0.4,
            "max_new_tokens": MAX_NEW_TOKENS,
            "test_used": False,
            "source_reasoning_used": False,
        },
        "generation": generation_audit,
        "scoring": score_audit,
        "direct": summarize(records, direct),
        "prompts": {},
    }
    for name in REASONING_PROMPTS:
        reasoning = scores[name]
        blend = sigmoid(0.6 * logit(direct) + 0.4 * logit(reasoning))
        report["prompts"][name] = {
            "reasoning": summarize(records, reasoning),
            "blend": summarize(records, blend),
            "blend_comparison_to_direct": comparison(
                records,
                direct,
                blend,
            ),
            "reply_audit": reply_audit(tokenizer, replies[name]),
        }
    report["selected_prompt"] = max(
        REASONING_PROMPTS,
        key=lambda name: report["prompts"][name]["blend"]["macro_auroc"],
    )

    output = records.drop(columns="messages").copy()
    output["direct_score"] = direct
    for name in REASONING_PROMPTS:
        output[f"{name}_reasoning_score"] = scores[name]
        output[f"{name}_reasoning_reply"] = replies[name]
    output.to_json(
        output_dir / "scores.jsonl",
        orient="records",
        lines=True,
        double_precision=15,
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
