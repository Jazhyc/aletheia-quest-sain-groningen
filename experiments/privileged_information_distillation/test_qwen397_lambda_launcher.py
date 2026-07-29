from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT
    / "experiments/privileged_information_distillation"
    / "run_qwen397_tvg_soft_teacher_lambda.sh"
)


def test_qwen397_launcher_adds_struct_only_dataset_filter() -> None:
    source = LAUNCHER.read_text()

    assert "+dataset_name_contains=varied-deception" in source
    assert "\n  dataset_name_contains=varied-deception" not in source
