from __future__ import annotations

import sys

import pandas as pd

from experiments.qwen_grpo_lora import evaluate_qwen_grpo_lora as generation_eval
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora_logits import parse_args


def test_exclude_reasoning_flag_defaults_off(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--adapter-dir", "adapter"],
    )

    assert parse_args().exclude_reasoning is False


def test_exclude_reasoning_flag_can_be_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--adapter-dir", "adapter", "--exclude-reasoning"],
    )

    assert parse_args().exclude_reasoning is True


def test_load_split_resolves_label_paths_from_explicit_project_root(
    monkeypatch,
    tmp_path,
) -> None:
    splits_dir = tmp_path / "project" / "dev_splits"
    observed = {}

    def fake_load_split_config(path, base):
        observed["path"] = path
        observed["base"] = base
        return [generation_eval.DatasetConfig(name="dataset", labels_uri="labels.csv")]

    monkeypatch.setattr(generation_eval, "load_split_config", fake_load_split_config)
    monkeypatch.setattr(
        generation_eval,
        "load_labels",
        lambda config: pd.DataFrame({"index": [1], "label": [0]}),
    )
    monkeypatch.setattr(
        generation_eval,
        "load_examples_for_labels",
        lambda *args, **kwargs: pd.DataFrame(
            {"dataset": ["dataset"], "index": [1], "label": [0], "prompt": ["prompt"]}
        ),
    )

    records = generation_eval.load_split(
        "validation",
        splits_dir,
        prompt_template="prompt",
        tokenizer=object(),
        max_prompt_chars=3000,
        context_truncation="tail",
        include_reasoning=False,
        reasoning_max_chars=0,
        reasoning_truncation="tail",
        enable_thinking=False,
    )

    assert observed == {
        "path": splits_dir / "dry.validation.yaml",
        "base": tmp_path / "project",
    }
    assert records.dataset_names == ["dataset"]
