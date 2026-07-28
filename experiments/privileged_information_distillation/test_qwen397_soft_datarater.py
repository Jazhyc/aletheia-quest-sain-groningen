from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT
    / "experiments/privileged_information_distillation"
    / "submit_qwen397_soft_datarater_full.sh"
)


def compose_config(name: str):
    with initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "configs").resolve()),
    ):
        return compose(config_name=name)


def test_soft_datarater_configs_preserve_identity_bce_and_fixed_compute() -> None:
    manifests = {
        "random50": "random_keep50.jsonl",
        "loss50": "loss_keep50.jsonl",
        "dot50": "dot_keep50.jsonl",
    }
    for selector, manifest in manifests.items():
        cfg = compose_config(
            f"pid_qwen397_soft_datarater_{selector}_fixed90_v1"
        )
        assert cfg.student.soft_teacher_artifact.endswith(
            "qwen35_397b_fp8_nothink_truth_value_binary_logit_v1/"
            "train/soft_targets.jsonl"
        )
        assert cfg.student.selection_manifest.endswith(manifest)
        assert cfg.student.training.max_steps == 90
        assert cfg.student.training.soft_loss_type == "bce"
        assert cfg.student.training.soft_target_logit_center == 0.0
        assert cfg.student.training.soft_target_logit_scale == 1.0
        assert cfg.student.training.completion_loss_weight == 0.0
        assert cfg.student.training.direct_loss_weight == 0.0
        assert cfg.student.training.pairwise_loss_weight == 0.0
        assert cfg.student.training.soft_loss_weight == 1.0


def test_soft_datarater_launcher_freezes_score_train_and_order_controls() -> None:
    source = LAUNCHER.read_text()

    assert 'sha256sum -c "${CACHE_ROOT}/SHA256SUMS"' in source
    assert "--objective soft_binary" in source
    assert "--finite-difference-epsilon 0.1" in source
    assert "--keep-fractions 0.5" in source
    assert '--dependency="afterok:${SCORE}"' in source
    assert "TORCHDYNAMO_DISABLE=1" in source
    assert 'TRAIN_DEPENDENCY="afterok:${RANDOM_JOB}:${LOSS_JOB}:${DOT_JOB}"' in source
    assert "--run-name validation_datarater_forward_v1" in source
    assert "--run-name validation_datarater_reverse_v1" in source
    assert "--continuous-margin-condition direct" in source
