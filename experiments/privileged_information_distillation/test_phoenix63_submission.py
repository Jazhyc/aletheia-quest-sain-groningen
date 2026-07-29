from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.privileged_information_distillation.upload_qwen397_tvg_adapter import (
    ADAPTER_DIR,
    REPOSITORY,
    validate_adapter,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "submission/phoenix_wright_v6_3.ipynb"
BUNDLED_ADAPTER = ROOT / "submission/phoenix_wright_adapters/main"


def notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_phoenix63_is_the_only_submission_notebook() -> None:
    notebooks = sorted((ROOT / "submission").glob("*.ipynb"))
    assert notebooks == [NOTEBOOK]


def test_phoenix63_changes_only_to_qwen397_binary_soft_main_adapter() -> None:
    source = notebook_source()
    assert "# Phoenix Wright v6.3" in source
    assert (
        "method=phoenix_wright_v6.3 "
        "phoenix4_renderer_qwen397_soft_ep2_specialists"
    ) in source
    assert f'"{REPOSITORY}"' in source
    assert "aletheias-phoenix-v5-qwen27-tvg-soft-r16" not in source
    assert 'SUBMISSION_THRESHOLD = float(os.environ.get("PHOENIX_THRESHOLD", "0.5"))' in source
    assert "Use 0 for not deceptive and 1 for deceptive." not in source
    assert "Output exactly one line: Prediction:<0 or 1>" in source
    assert "reasoning_summary" not in source


def test_phoenix63_adapter_is_canonical_float32_rank16() -> None:
    validation = validate_adapter(ADAPTER_DIR)
    assert validation["tensor_count"] == 256
    assert validation["dtypes"] == ["torch.float32"]


def test_bundled_main_is_the_exact_phoenix63_adapter() -> None:
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        trained = hashlib.sha256((ADAPTER_DIR / filename).read_bytes()).hexdigest()
        bundled = hashlib.sha256((BUNDLED_ADAPTER / filename).read_bytes()).hexdigest()
        assert bundled == trained
