from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]


def compose_config():
    with initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "configs").resolve()),
    ):
        return compose(config_name="pid_blind_teacher_material_rank16_full_v1")


def test_blind_reasoning_full_uses_every_varied_row_without_label_filtering() -> None:
    cfg = compose_config()

    assert cfg.teacher.uses_ground_truth is False
    assert cfg.teacher.dataset_name_contains == "varied-deception"
    assert cfg.teacher.selection_manifest is None
    assert cfg.teacher.limit is None
    assert cfg.teacher.limit_per_label is None
    assert cfg.teacher.max_tokens == 4096
    assert "__GROUND_TRUTH__" not in cfg.teacher.prompt

    assert cfg.student.dataset_name_contains == "varied-deception"
    assert cfg.student.selection_manifest is None
    assert cfg.student.train_fraction == 1.0
    assert cfg.student.require_teacher_label_match is False
    assert cfg.student.target_mode == "teacher"


def test_blind_reasoning_full_matches_selected_rank16_student_recipe() -> None:
    cfg = compose_config()

    assert cfg.student.model == "Qwen/Qwen3.5-9B"
    assert cfg.student.lora.r == 16
    assert cfg.student.lora.alpha == 32
    assert cfg.student.lora.dropout == 0.0
    assert cfg.student.training.optimizer == "adamw"
    assert cfg.student.training.learning_rate == 5.0e-5
    assert cfg.student.training.num_train_epochs == 1.0
    assert cfg.student.training.per_device_train_batch_size == 2
    assert cfg.student.training.gradient_accumulation_steps == 16
