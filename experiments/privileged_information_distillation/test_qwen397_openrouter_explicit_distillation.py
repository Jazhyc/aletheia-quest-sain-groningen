from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT
    / "experiments/privileged_information_distillation"
    / "submit_qwen397_openrouter_explicit_tvg_soft_distillation.sh"
)
TEACHER_RUNNER = (
    ROOT
    / "experiments/privileged_information_distillation"
    / "run_qwen397_openrouter_explicit_soft_teacher.sh"
)


def compose_config():
    with initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "configs").resolve()),
    ):
        return compose(
            config_name=(
                "pid_qwen397_openrouter_explicit_tvg_binary_soft_distillation_v1"
            )
        )


def test_openrouter_explicit_student_is_matched_to_new_cache_and_prompt() -> None:
    cfg = compose_config()

    assert (
        cfg.method
        == "qwen9b_qwen397_openrouter_explicit_tvg_binary_softonly_varied_v1"
    )
    assert cfg.teacher.artifact.endswith(
        "qwen35_397b_openrouter_nothink_tvg_binary_logit_explicit_digits_v1/"
        "train/student_rows.jsonl"
    )
    assert cfg.student.soft_teacher_artifact.endswith(
        "qwen35_397b_openrouter_nothink_tvg_binary_logit_explicit_digits_v1/"
        "train/soft_targets.jsonl"
    )
    assert "Use 0 for not deceptive and 1 for deceptive." in cfg.student.prompt
    assert cfg.student.model == "Qwen/Qwen3.5-9B"
    assert cfg.student.dataset_name_contains == "varied-deception"
    assert cfg.student.target_mode == "prediction_only"
    assert cfg.student.override_cached_prompt is True
    assert cfg.student.training.soft_loss_weight == 1.0
    assert cfg.student.training.completion_loss_weight == 0.0


def test_openrouter_explicit_launcher_validates_cache_and_chains_eval() -> None:
    source = LAUNCHER.read_text()

    assert 'sha256sum -c "${CACHE_ROOT}/SHA256SUMS"' in source
    assert 'if [[ "${rows}" -ne 2880 ]]' in source
    assert "--config-name \"${CONFIG}\"" in source
    assert '--dependency="afterok:${STUDENT}"' in source
    assert "--verify-lora-effect" in source
    assert "--continuous-margin-condition direct" in source


def test_openrouter_teacher_runner_freezes_selected_provenance() -> None:
    source = TEACHER_RUNNER.read_text()

    assert "--model qwen/qwen3.5-397b-a17b" in source
    assert "--prompt-variant explicit_digits" in source
    assert "--dataset-name-contains varied-deception" in source
    assert "--provider-only Alibaba" in source
    assert "--no-allow-fallbacks" in source
    assert "--expected-rows 2880" in source
    assert "sha256sum -c" in source
