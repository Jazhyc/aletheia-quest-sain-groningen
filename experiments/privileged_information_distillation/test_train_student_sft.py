import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.train_student_sft import load_records
from experiments.qwen_grpo_lora.run_qwen_grpo_lora import (
    MuonAdamW,
    muon_adamw_param_groups,
)


def test_load_records_filters_by_dataset_name(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    records = [
        {
            "dataset": "dev-instructed-deception-model",
            "parse_error": False,
            "label_match": True,
            "student_target": "Prediction:0",
        },
        {
            "dataset": "dev-varied-deception-model",
            "parse_error": False,
            "label_match": True,
            "student_target": "Prediction:1",
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    selected = load_records(path, dataset_name_contains="varied-deception")

    assert [record["dataset"] for record in selected] == ["dev-varied-deception-model"]


def test_load_records_without_filter_preserves_all_usable_rows(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    record = {
        "dataset": "dev-instructed-deception-model",
        "parse_error": False,
        "label_match": True,
        "student_target": "Prediction:0",
    }
    path.write_text(json.dumps(record) + "\n")

    assert load_records(path) == [record]


def test_muon_groups_and_updates_only_trainable_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
    model[1].weight.requires_grad_(False)
    groups = muon_adamw_param_groups(model, weight_decay=0.0)

    algorithms = {group["algorithm"] for group in groups}
    assert algorithms == {"muon", "adamw"}
    assert all(model[1].weight is not param for group in groups for param in group["params"])

    optimizer = MuonAdamW(groups, lr=1e-6, muon_lr=3e-5)
    before = model[0].weight.detach().clone()
    model[0](torch.ones(1, 3)).sum().backward()
    optimizer.step()

    assert not torch.equal(before, model[0].weight)
