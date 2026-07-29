from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.privileged_information_distillation.upload_kimi_k3_tvg_adapter import (
    REPOSITORY,
    validate_adapter,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "submission/phoenix_wright_v8_0.ipynb"
BUNDLED_ADAPTER = ROOT / "submission/phoenix_wright_adapters/main"
REMOVED_INTENT_ADAPTER = (
    ROOT / "submission/phoenix_wright_adapters/resolved_intent"
)


def notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_phoenix80_is_the_only_submission_notebook() -> None:
    notebooks = sorted((ROOT / "submission").glob("*.ipynb"))
    assert notebooks == [NOTEBOOK]


def test_phoenix80_uses_one_kimi_student_pass_without_intent_adapter() -> None:
    source = notebook_source()
    assert "# Phoenix Wright v8.0" in source
    assert "method=phoenix_wright_v8.0 kimi_k3_distilled_routes" in source
    assert f'"{REPOSITORY}"' in source
    assert "aletheias-phoenix-v6-3-qwen397-tvg-soft-r16-ep2" not in source
    assert "aletheias-phoenix-v5-qwen27-tvg-soft-r16" not in source
    assert "ACTION_ADAPTER_REPO" not in source
    assert 'kind == "harm"' not in source
    assert 'kind == "action"' not in source
    assert "score_distilled_routes(messages, harm_flags, action_flags)" in source
    assert "build_primary_route_flags" in source
    assert "build_epistemic_empty_prompt" in source
    assert "build_action_empty_prompt" in source
    assert "build_intent_empty_prompt" not in source
    assert "blend_intent_route_scores" not in source
    assert "PHOENIX_INTENT_ADAPTER_REPO" not in source
    assert "aletheias-phoenix-v3-resolved-intent-r1" not in source
    assert not REMOVED_INTENT_ADAPTER.exists()
    assert (
        'SUBMISSION_THRESHOLD = float('
        'os.environ.get("PHOENIX_THRESHOLD", "0.5"))'
    ) in source
    assert "Use 0 for not deceptive and 1 for deceptive." not in source
    assert "Output exactly one line: Prediction:<0 or 1>" in source
    assert "model.generate" not in source


def test_phoenix80_main_adapter_is_canonical_float32_rank16() -> None:
    validation = validate_adapter(BUNDLED_ADAPTER)
    assert validation["tensor_count"] == 256
    assert validation["dtypes"] == ["torch.float32"]
    assert (
        validation["weight_sha256"]
        == "c5025a39dd05af16405c692a0c1b70657afd4f8e4a4e634bd789b5d67b4a9eb0"
    )


def test_bundled_main_config_matches_published_phoenix80_adapter() -> None:
    digest = hashlib.sha256(
        (BUNDLED_ADAPTER / "adapter_config.json").read_bytes()
    ).hexdigest()
    assert digest == "d9ba5c277d1843409fa8b4162fc219f992feeed80638dc32788927cf6fe178ae"
