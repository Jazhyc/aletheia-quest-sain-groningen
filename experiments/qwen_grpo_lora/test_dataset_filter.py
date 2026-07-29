import sys
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_qwen_grpo_lora import SplitRecords, filter_datasets


def records() -> SplitRecords:
    return SplitRecords(
        frame=pd.DataFrame({"dataset": ["dev-varied-deception-a", "dev-instructed-deception-b"]}),
        dataset_names=["dev-varied-deception-a", "dev-instructed-deception-b"],
    )


def test_filter_datasets_uses_literal_substring() -> None:
    filtered = filter_datasets(records(), "varied-deception")
    assert filtered.frame["dataset"].tolist() == ["dev-varied-deception-a"]
    assert filtered.dataset_names == ["dev-varied-deception-a"]


def test_filter_datasets_rejects_empty_selection() -> None:
    with pytest.raises(RuntimeError, match="no datasets"):
        filter_datasets(records(), "missing")


def test_q397_optimized_reasoning_config() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="qwen_grpo_lora_q397_optimized_reasoning")

    assert cfg.model == "Qwen/Qwen3.5-9B"
    assert str(cfg.init_adapter).endswith(
        "qwen9b_qwen397_tvg_soft_r16_lr5e5_ep2_v1/"
        "peft_path_migration/local_backups/adapter"
    )
    assert cfg.train_dataset_name_contains == "varied-deception"
    assert cfg.judge.include_reasoning is False
    assert "<reasoning>" in cfg.judge.prompt
    assert "Reason carefully" in cfg.judge.prompt
    assert "<reasoning>" not in cfg.inference.prompt
    assert cfg.inference.prompt.endswith("Output exactly one line: Prediction:<0 or 1>")
    assert cfg.lora.r == 16
    assert cfg.lora.alpha == 32
    assert cfg.training.per_device_train_batch_size == 4
    assert cfg.training.num_generations == 4
    assert cfg.training.generation_batch_size == 32
    assert cfg.training.max_completion_length == 256
    assert cfg.evaluation.generate_completions is False
