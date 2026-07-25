from __future__ import annotations

import json
from pathlib import Path
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.select_hpkr_rank1_specialist import (
    compare,
    paired,
)
from experiments.liars_bench_hpkr_prompts.evaluate import PROMPTS


def metrics(
    ba: float,
    llama: float,
    mistral: float,
    qwen: float,
    parse: int = 0,
) -> dict:
    return {
        "balanced_accuracy": ba,
        "parse_errors": parse,
        "per_source_model": {
            "llama": {"balanced_accuracy": None, "accuracy": llama},
            "mistral": {
                "balanced_accuracy": mistral,
                "accuracy": mistral,
            },
            "qwen": {"balanced_accuracy": qwen, "accuracy": qwen},
        },
    }


def test_rank1_training_prompt_matches_frozen_epistemic_prompt() -> None:
    config = OmegaConf.load(
        ROOT / "configs/pid_liars_bench_hpkr_specialist_rank1_e1_lr5e5_v1.yaml"
    )
    assert config.student.prompt == PROMPTS["knowledge_report_type"]
    assert config.student.lora.r == 1
    assert config.student.lora.alpha == 2
    assert config.student.override_cached_prompt is True


def test_gate_requires_gain_and_every_source_floor() -> None:
    baseline = metrics(0.80, 0.90, 0.60, 0.90)
    passing = compare(
        metrics(0.84, 0.92, 0.64, 0.91, parse=1),
        baseline,
        minimum_gain=0.02,
        minimum_source_delta=-0.05,
        maximum_parse_error_increase=5,
    )
    assert passing["passes"]

    source_failure = compare(
        metrics(0.84, 0.92, 0.70, 0.84),
        baseline,
        minimum_gain=0.02,
        minimum_source_delta=-0.05,
        maximum_parse_error_increase=5,
    )
    assert not source_failure["passes"]


def test_paired_counts_fixes_and_breaks() -> None:
    baseline = [
        {"dataset": "d", "index": "a", "label": 1, "prediction": 0},
        {"dataset": "d", "index": "b", "label": 0, "prediction": 0},
        {"dataset": "d", "index": "c", "label": 1, "prediction": 1},
    ]
    candidate = json.loads(json.dumps(baseline))
    candidate[0]["prediction"] = 1
    candidate[2]["prediction"] = 0
    assert paired(candidate, baseline) == {
        "changes": 2,
        "fixes": 1,
        "breaks": 1,
    }
