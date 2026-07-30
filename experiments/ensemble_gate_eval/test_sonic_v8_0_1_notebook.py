"""Pin Sonic v8.0.1 as an adapter-only transfer probe from Sonic v8.0."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "official_submissions/sonic_v8_0/sonic_v8_0.ipynb"
NOTEBOOK = ROOT / "submission/sonic_v8_0_1.ipynb"

OLD_ADAPTER = "Jazhyc/aletheias-phoenix-v8-kimi-k3-tvg-soft-full-r16-ep2"
NEW_ADAPTER = "Jazhyc/aletheias-phoenix-v8-1-kimi-k3-liars-full-r16-ep2"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def code_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_submission_contains_exactly_one_notebook() -> None:
    assert list((ROOT / "submission").glob("*.ipynb")) == [NOTEBOOK]


def test_notebook_uses_phoenix81_adapter_and_version() -> None:
    notebook = load_notebook(NOTEBOOK)
    source = code_source(notebook)

    assert NEW_ADAPTER in source
    assert OLD_ADAPTER not in source
    assert "method = sonic_v8.0.1" in source
    assert "judge=v8.1 kimi_k3_liars" in source


def test_all_code_cells_parse() -> None:
    for cell in load_notebook(NOTEBOOK)["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_probe_gate_and_runtime_contract_remain_v80() -> None:
    source = code_source(load_notebook(NOTEBOOK))

    assert 'JUDGE_LOGIT_MEAN", "0.0"' in source
    assert 'JUDGE_LOGIT_SD", "5.638"' in source
    assert "BASE_CAP = 1.0 * _step_z" in source
    assert "MAX_CAP  = 4.0 * _step_z" in source
    assert "agreement = (judge_z * probe_z > 0)" in source
    assert 'PHOENIX_THRESHOLD' not in source
    assert 'SUBMISSION_THRESHOLD", "0.5"' in source
    assert 'PHOENIX_BATCH_SIZE", "48"' in source
    assert 'PHOENIX_MEDIUM_BATCH_SIZE", "32"' in source
    assert 'PHOENIX_LONG_BATCH_SIZE", "16"' in source
    assert '"logits_to_keep": 1' in source
    assert "model.generate" not in source


def test_only_documentation_adapter_and_logging_changed_from_v80() -> None:
    source = load_notebook(SOURCE)
    target = load_notebook(NOTEBOOK)
    expected = copy.deepcopy(source)

    replacements = {
        "# sonic v8.0 — dual probe + Phoenix v8 judge, retuned v4 sign gate\n":
            "# Sonic v8.0.1 — dual probe + Phoenix v8.1 judge, v8.0 sign-gate calibration\n",
        "v8.0 direct-margin judge (Kimi K3 distilled, r16 adapter, HP-KR + action routes).\n":
            "v8.1 direct-margin judge (Kimi K3 Liars-enriched, r16 adapter, HP-KR + action routes).\n",
        "Blended under the v4 sign gate with **retuned constants** for the v8 judge.\n":
            "Blended under the unchanged v8.0 sign gate and its frozen judge-scale constants.\n",
        "## v8.0 change from v6.1\n": "## v8.0.1 change from v8.0\n",
        "v6.1 used the v6.2 judge (multi-adapter, intent blending). v8.0 upgrades to\n":
            "Only the judge adapter changes: Phoenix v8.0's competition-only Kimi K3\n",
        "the Phoenix v8 judge (single adapter, Kimi K3 distilled, direct-margin).\n":
            "student is replaced by the Phoenix v8.1 competition-plus-Liars student.\n",
        "Gate constants are recomputed for the v8 judge's score distribution.\n":
            "The probe, routes, renderer, direct-margin readout, gate, and binary threshold\n"
            "are unchanged. `JUDGE_LOGIT_SD=5.638` remains the v8.0 calibration; this is an\n"
            "intentional uncalibrated leaderboard transfer probe for the v8.1 judge.\n",
        "(RTX 4090, Qwen3.5-9B + v8 adapter). Override via env var.\n":
            "(RTX 4090, Qwen3.5-9B + v8.0 adapter). It is intentionally reused without\n"
            "recalibration for the v8.1 adapter and can still be overridden via env var.\n",
        "print(f\"dataset = {DATASET_NAME}\")\n":
            "print(\"method = sonic_v8.0.1\")\nprint(f\"dataset = {DATASET_NAME}\")\n",
        "# Phoenix Wright v8.0 direct-margin judge — Kimi K3 distilled, single adapter.\n":
            "# Phoenix Wright v8.1 direct-margin judge — Kimi K3 Liars-enriched, single adapter.\n",
        OLD_ADAPTER: NEW_ADAPTER,
        "print(f\"judge=v8.0 kimi_k3 adapter={MAIN_ADAPTER_REPO} \"\n":
            "print(f\"judge=v8.1 kimi_k3_liars adapter={MAIN_ADAPTER_REPO} \"\n",
        "# v8.0: retuned for Phoenix v8 judge.  When the judge and probe agree on\n":
            "# Sonic v8.0.1 intentionally retains v8.0's Phoenix-judge calibration. When\n"
            "# the v8.1 judge and probe agree on\n",
        "            f\"gate: v8.0 sign cap (retuned for v8 judge), {len(final_scores)} rows, \"\n":
            "            f\"gate: v8.0.1 sign cap (v8.0 calibration; uncalibrated for v8.1 judge), \"\n"
            "            f\"{len(final_scores)} rows, \"\n",
    }

    for cell in expected["cells"]:
        updated = []
        for line in cell["source"]:
            for old, new in replacements.items():
                line = line.replace(old, new)
            updated.extend(new_line for new_line in line.splitlines(keepends=True))
        cell["source"] = updated

    assert target == expected
