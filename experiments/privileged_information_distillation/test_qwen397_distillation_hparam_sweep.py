from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT
    / "experiments/privileged_information_distillation"
    / "run_qwen397_distillation_hparam_sweep_lambda.sh"
)
SUMMARY_PATH = (
    ROOT
    / "experiments/privileged_information_distillation"
    / "summarize_qwen397_distillation_sweep.py"
)


def load_summary_module():
    spec = importlib.util.spec_from_file_location("q397_sweep_summary", SUMMARY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_freezes_cache_and_rank16_grid() -> None:
    source = LAUNCHER.read_text()

    assert 'sha256sum -c "${CACHE_ROOT}/SHA256SUMS"' in source
    assert 'if [[ "${rows}" -ne 2880 ]]' in source
    assert "learning_rates=(1e-5 2e-5 5e-5 1e-4)" in source
    assert "epochs=(0.5 1.0 2.0)" in source
    assert "train_adapter \\\n      16 32" in source
    assert "--continuous-margin-condition direct" in source
    assert "--verify-lora-effect" in source
    assert "flash-linear-attention==0.5.2" in source
    assert 'MICRO_BATCH="${Q397_MICRO_BATCH:-8}"' in source
    assert 'GRADIENT_ACCUMULATION="${Q397_GRADIENT_ACCUMULATION:-4}"' in source
    assert '"student.training.torch_compile=false"' in source


def test_launcher_uses_selected_rank16_response_surface_for_rank24() -> None:
    source = LAUNCHER.read_text()

    assert "--best-shell" in source
    assert "train_adapter \\\n    24 48" in source
    assert 'method_name 24 "${learning_rate_names[$lr_index]}" "${best_epoch_name}"' in source
    assert "migrate_qwen35_peft_paths.py" in source


def test_summary_method_pattern_captures_sweep_coordinates() -> None:
    module = load_summary_module()
    match = module.METHOD_RE.fullmatch(
        "qwen9b_qwen397_tvg_soft_r24_lr5e5_ep05_v1"
    )

    assert match is not None
    assert match.group("rank") == "24"
    assert match.group("lr_name") == "5e5"
    assert match.group("epoch_name") == "ep05"
