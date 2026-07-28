from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT
    / "experiments/privileged_information_distillation"
    / "submit_qwen397_tvg_soft_distillation.sh"
)
ABLATION_LAUNCHER = (
    ROOT
    / "experiments/privileged_information_distillation"
    / "submit_qwen397_tvg_objective_ablation.sh"
)


def compose_config():
    with initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "configs").resolve()),
    ):
        return compose(config_name="pid_qwen397_tvg_binary_soft_distillation_v1")


def test_qwen397_soft_student_matches_frozen_pure_boundary_recipe() -> None:
    cfg = compose_config()

    assert cfg.method == "qwen9b_qwen397_tvg_binary_softonly_varied_v1"
    assert cfg.teacher.artifact.endswith(
        "qwen35_397b_fp8_nothink_truth_value_binary_logit_v1/train/student_rows.jsonl"
    )
    assert cfg.student.soft_teacher_artifact.endswith(
        "qwen35_397b_fp8_nothink_truth_value_binary_logit_v1/train/soft_targets.jsonl"
    )
    assert cfg.student.model == "Qwen/Qwen3.5-9B"
    assert cfg.student.model_loader == "causal_lm"
    assert cfg.student.dataset_name_contains == "varied-deception"
    assert cfg.student.target_mode == "prediction_only"
    assert cfg.student.override_cached_prompt is True
    assert cfg.student.lora.r == 16
    assert cfg.student.lora.alpha == 32
    assert cfg.student.lora.exclude_modules == (
        ".*(visual|vision_tower|merger|patch_embed).*"
    )
    assert cfg.student.training.optimizer == "adamw"
    assert cfg.student.training.learning_rate == 5.0e-5
    assert cfg.student.training.num_train_epochs == 1.0
    assert cfg.student.training.per_device_train_batch_size == 2
    assert cfg.student.training.gradient_accumulation_steps == 16
    assert cfg.student.training.completion_loss_weight == 0.0
    assert cfg.student.training.direct_loss_weight == 0.0
    assert cfg.student.training.pairwise_loss_weight == 0.0
    assert cfg.student.training.soft_loss_weight == 1.0


def test_qwen397_launcher_validates_cache_and_chains_direct_margin_eval() -> None:
    source = LAUNCHER.read_text()

    assert 'sha256sum -c "${CACHE_ROOT}/SHA256SUMS"' in source
    assert 'if [[ "${rows}" -ne 2880 ]]' in source
    assert "--config-name \"${CONFIG}\"" in source
    assert "QWEN35_CANONICALIZE_ADAPTER=${ROOT}/${ADAPTER}" in source
    assert '--dependency="afterok:${STUDENT}"' in source
    assert "--verify-lora-effect" in source
    assert "--continuous-margin-condition direct" in source


def test_qwen397_objective_configs_change_only_soft_target_geometry() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "configs").resolve()),
    ):
        bce = compose(
            config_name="pid_qwen397_tvg_binary_soft_distillation_zscore_bce_v1"
        )
        huber = compose(
            config_name="pid_qwen397_tvg_binary_soft_distillation_zscore_huber_v1"
        )

    assert bce.student.training.soft_loss_type == "bce"
    assert huber.student.training.soft_loss_type == "huber"
    for cfg in (bce, huber):
        assert cfg.student.training.soft_target_logit_center == -2.291015627315462
        assert cfg.student.training.soft_target_logit_scale == 6.31331511070083
        assert cfg.student.training.soft_huber_delta == 1.0
        assert cfg.student.training.completion_loss_weight == 0.0
        assert cfg.student.training.direct_loss_weight == 0.0
        assert cfg.student.training.pairwise_loss_weight == 0.0
        assert cfg.student.training.soft_loss_weight == 1.0


def test_qwen397_objective_launcher_submits_both_dependent_evaluations() -> None:
    source = ABLATION_LAUNCHER.read_text()

    assert 'sha256sum -c "${CACHE_ROOT}/SHA256SUMS"' in source
    assert "pid_qwen397_tvg_binary_soft_distillation_zscore_bce_v1" in source
    assert "pid_qwen397_tvg_binary_soft_distillation_zscore_huber_v1" in source
    assert '--dependency="afterok:${student}"' in source
    assert "--verify-lora-effect" in source
    assert "--continuous-margin-condition direct" in source
