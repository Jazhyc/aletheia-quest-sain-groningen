from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHOD_DIR = ROOT / "experiments/privileged_information_distillation"
LAUNCHER = METHOD_DIR / "run_kimi_k3_distillation_hparam_sweep_lambda.sh"
SUMMARY_PATH = METHOD_DIR / "summarize_kimi_k3_distillation_sweep.py"


def load_summary_module():
    spec = importlib.util.spec_from_file_location("kimi_sweep_summary", SUMMARY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_freezes_cache_and_matched_rank16_grid() -> None:
    source = LAUNCHER.read_text()

    assert 'sha256sum -c "${CACHE_ROOT}/SHA256SUMS"' in source
    assert 'if [[ "${rows}" -ne 2880 ]]' in source
    assert "learning_rates=(1e-5 2e-5 5e-5 1e-4)" in source
    assert "epochs=(0.5 1.0 2.0)" in source
    assert "--continuous-margin-condition direct" in source
    assert "--verify-lora-effect" in source
    assert "flash-linear-attention==0.5.2" in source
    assert 'MICRO_BATCH="${KIMI_MICRO_BATCH:-8}"' in source
    assert 'GRADIENT_ACCUMULATION="${KIMI_GRADIENT_ACCUMULATION:-4}"' in source
    assert '"student.training.torch_compile=false"' in source
    assert "migrate_qwen35_peft_paths.py" in source


def test_summary_method_pattern_captures_sweep_coordinates() -> None:
    module = load_summary_module()
    match = module.METHOD_RE.fullmatch(
        "qwen9b_kimi_k3_openrouter_tvg_soft_r16_lr5e5_ep05_v1"
    )

    assert match is not None
    assert match.group("rank") == "16"
    assert match.group("lr_name") == "5e5"
    assert match.group("epoch_name") == "ep05"
