from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "pid_kimi_k3_openrouter_tvg_binary_soft_distillation_v1.yaml"
FULL_CONFIG = (
    ROOT
    / "configs"
    / "pid_kimi_k3_openrouter_tvg_binary_soft_full_r16_ep2_v1.yaml"
)
RANK32_CONFIG = (
    ROOT
    / "configs"
    / "pid_kimi_k3_openrouter_tvg_binary_soft_full_r32a64_ep2_bf16_v1.yaml"
)
TEACHER = (
    ROOT
    / "experiments"
    / "privileged_information_distillation"
    / "run_kimi_k3_openrouter_soft_teacher.sh"
)
INSTRUCTED_TEACHER = (
    ROOT
    / "experiments"
    / "privileged_information_distillation"
    / "run_kimi_k3_openrouter_instructed_soft_teacher.sh"
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
FULL_CACHE = (
    ROOT
    / "experiments"
    / "privileged_information_distillation"
    / "prepare_kimi_k3_full_soft_cache.sh"
)
FULL_LAMBDA = (
    ROOT
    / "experiments"
    / "privileged_information_distillation"
    / "run_kimi_k3_full_distillation_lambda.sh"
)
RANK32_LAMBDA = (
    ROOT
    / "experiments"
    / "privileged_information_distillation"
    / "run_kimi_k3_full_r32a64_bf16_lambda.sh"
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


def test_kimi_instructed_teacher_is_pinned_and_builds_identity_targets() -> None:
    source = INSTRUCTED_TEACHER.read_text()

    assert 'CONCURRENCY="${KIMI_CONCURRENCY:-64}"' in source
    assert "--model moonshotai/kimi-k3" in source
    assert "--provider-only Fireworks" in source
    assert "--no-allow-fallbacks" in source
    assert "--dataset-name-contains instructed-deception" in source
    assert "--kind binary_identity" in source
    assert "--expected-rows 3693" in source


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


def test_kimi_full_config_keeps_rank16_two_epoch_recipe() -> None:
    config = yaml.safe_load(FULL_CONFIG.read_text())

    assert (
        config["defaults"][0]
        == "pid_kimi_k3_openrouter_tvg_binary_soft_distillation_v1"
    )
    assert config["method"] == (
        "qwen9b_kimi_k3_openrouter_tvg_soft_full_r16_lr5e5_ep2_v1"
    )
    assert config["student"]["dataset_name_contains"] is None
    assert config["student"]["training"]["num_train_epochs"] == 2.0
    assert "binary_logit_full_v1" in config["teacher"]["artifact"]
    assert "binary_logit_full_v1" in config["student"]["soft_teacher_artifact"]


def test_kimi_full_cache_merges_both_frozen_sources() -> None:
    source = FULL_CACHE.read_text()

    assert "kimi_k3_fireworks_nothink_tvg_binary_logit_v1" in source
    assert "binary_logit_instructed_v1" in source
    assert "--additional-input" in source
    assert '--dataset-name-contains ""' in source
    assert "--expected-rows 6573" in source


def test_kimi_full_lambda_runner_uses_selected_h100_recipe() -> None:
    source = FULL_LAMBDA.read_text()

    assert "flash-linear-attention==0.5.2" in source
    assert 'MICRO_BATCH="${KIMI_MICRO_BATCH:-8}"' in source
    assert 'GRADIENT_ACCUMULATION="${KIMI_GRADIENT_ACCUMULATION:-4}"' in source
    assert "pid_kimi_k3_openrouter_tvg_binary_soft_full_r16_ep2_v1" in source
    assert "expected 6573 rows" in source
    assert "--split validation" in source
    assert "--split test" not in source
    assert "--continuous-margin-condition direct" in source


def test_kimi_rank32_config_preserves_lora_scale_and_two_epochs() -> None:
    config = yaml.safe_load(RANK32_CONFIG.read_text())

    assert (
        config["defaults"][0]
        == "pid_kimi_k3_openrouter_tvg_binary_soft_full_r16_ep2_v1"
    )
    assert config["student"]["lora"] == {"r": 32, "alpha": 64}
    assert config["student"]["training"]["num_train_epochs"] == 2.0
    assert config["student"]["output_dir"].endswith("/adapter_fp32")


def test_kimi_rank32_runner_evaluates_exact_bf16_package() -> None:
    source = RANK32_LAMBDA.read_text()

    assert 'MICRO_BATCH="${KIMI_MICRO_BATCH:-8}"' in source
    assert 'GRADIENT_ACCUMULATION="${KIMI_GRADIENT_ACCUMULATION:-4}"' in source
    assert "--dtype bfloat16" in source
    assert "expected all BF16 LoRA tensors" in source
    assert '--adapter-dir "${ADAPTER}"' in source
    assert "--split validation" in source
    assert "--split test" not in source
    assert "--continuous-margin-condition direct" in source
