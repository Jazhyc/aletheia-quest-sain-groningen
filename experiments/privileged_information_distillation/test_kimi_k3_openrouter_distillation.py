from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "pid_kimi_k3_openrouter_tvg_binary_soft_distillation_v1.yaml"
TEACHER = (
    ROOT
    / "experiments"
    / "privileged_information_distillation"
    / "run_kimi_k3_openrouter_soft_teacher.sh"
)
SUBMIT = (
    ROOT
    / "experiments"
    / "privileged_information_distillation"
    / "submit_kimi_k3_openrouter_soft_distillation.sh"
)
LAMBDA = (
    ROOT
    / "experiments"
    / "privileged_information_distillation"
    / "run_kimi_k3_distillation_lambda.sh"
)


def test_kimi_config_uses_matched_binary_soft_recipe() -> None:
    config = yaml.safe_load(CONFIG.read_text())

    assert config["defaults"][0] == "pid_qwen397_tvg_binary_soft_distillation_v1"
    assert config["method"] == "qwen9b_kimi_k3_openrouter_tvg_soft_r16_lr5e5_ep2_v1"
    assert config["student"]["training"]["num_train_epochs"] == 2.0
    assert "kimi_k3_fireworks" in config["teacher"]["artifact"]
    assert "kimi_k3_fireworks" in config["student"]["soft_teacher_artifact"]


def test_kimi_teacher_is_pinned_and_builds_identity_targets() -> None:
    source = TEACHER.read_text()

    assert "--model moonshotai/kimi-k3" in source
    assert "--provider-only Fireworks" in source
    assert "--no-allow-fallbacks" in source
    assert "--dataset-name-contains varied-deception" in source
    assert "--kind binary_identity" in source
    assert "--expected-rows 2880" in source


def test_kimi_student_only_schedules_validation() -> None:
    source = SUBMIT.read_text()

    assert "--split validation" in source
    assert "--split test" not in source
    assert "--continuous-margin-condition direct" in source
    assert "--verify-lora-effect" in source


def test_kimi_lambda_runner_uses_selected_h100_recipe() -> None:
    source = LAMBDA.read_text()

    assert "flash-linear-attention==0.5.2" in source
    assert 'MICRO_BATCH="${KIMI_MICRO_BATCH:-8}"' in source
    assert 'GRADIENT_ACCUMULATION="${KIMI_GRADIENT_ACCUMULATION:-4}"' in source
    assert "--config-name \"${BASE_CONFIG}\"" in source
    assert "--split validation" in source
    assert "--split test" not in source
    assert "--continuous-margin-condition direct" in source
