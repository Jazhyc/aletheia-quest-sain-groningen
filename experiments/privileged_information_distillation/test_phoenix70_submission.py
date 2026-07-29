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
NOTEBOOK = ROOT / "submission/phoenix_wright_v7_0.ipynb"
BUNDLED_ADAPTER = ROOT / "submission/phoenix_wright_adapters/main"


def notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_phoenix70_is_the_only_submission_notebook() -> None:
    notebooks = sorted((ROOT / "submission").glob("*.ipynb"))
    assert notebooks == [NOTEBOOK]


def test_phoenix70_uses_q397_for_every_primary_route_and_rank1_intent() -> None:
    source = notebook_source()
    assert "# Phoenix Wright v7.0" in source
    assert (
        "method=phoenix_wright_v7.0 "
        "q397_consolidated_routes_rank1_intent"
    ) in source
    assert f'"{REPOSITORY}"' in source
    assert "aletheias-phoenix-v5-qwen27-tvg-soft-r16" not in source
    assert "ACTION_ADAPTER_REPO" not in source
    assert 'kind == "harm"' not in source
    assert 'kind == "action"' not in source
    assert "score_distilled_routes(messages, harm_flags, action_flags)" in source
    assert "build_epistemic_empty_prompt" in source
    assert "build_action_empty_prompt" in source
    assert "build_intent_empty_prompt" in source
    assert "blend_intent_route_scores" in source
    assert (
        'SUBMISSION_THRESHOLD = float('
        'os.environ.get("PHOENIX_THRESHOLD", "0.5"))'
    ) in source
    assert "Use 0 for not deceptive and 1 for deceptive." not in source
    assert "Output exactly one line: Prediction:<0 or 1>" in source
    assert "model.generate" not in source


def test_phoenix70_main_adapter_is_canonical_float32_rank16() -> None:
    validation = validate_adapter(ADAPTER_DIR)
    assert validation["tensor_count"] == 256
    assert validation["dtypes"] == ["torch.float32"]


def test_bundled_main_is_the_exact_phoenix70_adapter() -> None:
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        trained = hashlib.sha256((ADAPTER_DIR / filename).read_bytes()).hexdigest()
        bundled = hashlib.sha256((BUNDLED_ADAPTER / filename).read_bytes()).hexdigest()
        assert bundled == trained
