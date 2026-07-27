from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]


def test_luna_teacher_student_config_matches_frozen_recipe() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "configs"),
    ):
        cfg = compose(config_name="pid_teacher_gpt56_luna_medium_varied_v1")

    assert cfg.teacher.model == "openai/gpt-5.6-luna"
    assert cfg.teacher.reasoning_effort == "medium"
    assert cfg.teacher.dataset_name_contains == "varied-deception"
    assert str(cfg.teacher.artifact).endswith(
        "qwen9b_privileged_gpt56_luna_medium_tvg_variedonly_v1/generations.jsonl"
    )
    assert cfg.student.model == "Qwen/Qwen3.5-9B"
    assert cfg.student.dataset_name_contains == "varied-deception"
    assert cfg.student.lora.r == 16
    assert cfg.student.lora.alpha == 32
    assert cfg.student.training.optimizer == "adamw"
    assert cfg.student.training.learning_rate == 5.0e-5
    assert cfg.student.training.num_train_epochs == 1.0
    assert cfg.student.training.per_device_train_batch_size == 2
    assert cfg.student.training.gradient_accumulation_steps == 16
