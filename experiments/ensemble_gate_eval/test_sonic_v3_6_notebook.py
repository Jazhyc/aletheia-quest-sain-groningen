"""Pin sonic v3.6: v3.5 with MAX_CAP raised from 4 to 12 judge steps.

v3.6 moves exactly one constant, so these tests are mostly about what must NOT
have moved. The gate shape, BASE_CAP, PROBE_GAIN, the probe standardization and
both thresholds are v3.5's; the pipeline cells must be byte-identical.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import build_sonic_v3_6_notebook as builder  # noqa: E402

SOURCE = REPO_ROOT / "legacy_submissions/sonic_v3_5.ipynb"
CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_6_constants.json"
V3_5_CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_5_constants.json"

SCORING_CELL = 12
PIPELINE_CELLS = tuple(range(1, 12))
UNCHANGED = ("JUDGE_MARGIN_SD", "BASE_CAP", "PROBE_GAIN", "PROBE_LOGIT_MEAN",
             "PROBE_LOGIT_SD", "DEFAULT_PROBE_MEAN", "DEFAULT_PROBE_SD",
             "THRESHOLD_SCORE", "THRESHOLD_PROBE_ONLY")
BATCH_STATISTICS = ("_rank01", "np.quantile", "np.median", "roc_auc_score",
                    "argsort(probe", ".mean()", ".std()")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    if not SOURCE.exists() or not CONSTANTS.exists():
        pytest.skip("v3.5 notebook or v3.6 constants not present")
    output = tmp_path_factory.mktemp("sonic_v3_6") / "sonic_v3_6.ipynb"
    builder.main(["--source", str(SOURCE), "--output", str(output)])
    return json.loads(output.read_text())


@pytest.fixture(scope="module")
def source_notebook() -> dict:
    if not SOURCE.exists():
        pytest.skip(f"{SOURCE} not present")
    return json.loads(SOURCE.read_text())


def cell_text(notebook: dict, i: int) -> str:
    return "".join(notebook["cells"][i]["source"])


def literals(text: str) -> dict:
    """Every top-level `NAME = <literal>` assignment in a cell."""
    out = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, rhs = stripped.partition("=")
        name = name.strip()
        if not name.isupper() or not name.replace("_", "").isalnum():
            continue
        try:
            out[name] = ast.literal_eval(rhs.strip())
        except (ValueError, SyntaxError):
            continue
    return out


def test_cell_count(built):
    assert len(built["cells"]) == 13


def test_pipeline_cells_are_byte_identical(built, source_notebook):
    """Only the header and the scoring cell may differ from v3.5."""
    for i in PIPELINE_CELLS:
        assert cell_text(built, i) == cell_text(source_notebook, i), f"cell {i} moved"


def test_max_cap_is_twelve_steps(built):
    got = literals(cell_text(built, SCORING_CELL))
    step = json.loads(CONSTANTS.read_text())["judge_step_z"]
    assert math.isclose(got["MAX_CAP"], 12 * step, rel_tol=1e-12)
    assert math.isclose(got["MAX_CAP"], 1.2502551166278297, rel_tol=1e-12)


def test_base_cap_and_everything_else_unchanged(built, source_notebook):
    new = literals(cell_text(built, SCORING_CELL))
    old = literals(cell_text(source_notebook, SCORING_CELL))
    for name in UNCHANGED:
        assert new[name] == old[name], f"{name} moved and must not have"
    assert new["BASE_CAP"] * 2 < new["MAX_CAP"], "MAX_CAP must exceed BASE_CAP"


def test_old_cap_is_gone(built):
    executable = "\n".join(line for line in cell_text(built, SCORING_CELL).splitlines()
                           if not line.strip().startswith("#"))
    assert "0.41675170554260993" not in executable


def test_gate_is_still_a_sign_test(built):
    text = cell_text(built, SCORING_CELL)
    assert "(combined * probe_z > 0)" in text
    executable = "\n".join(line for line in text.splitlines()
                           if not line.strip().startswith("#"))
    assert "AGREEMENT_SCALE" not in executable


def test_no_batch_statistics_in_scoring(built):
    """The scoring path must not read the batch -- every constant frozen offline."""
    executable = "\n".join(line for line in cell_text(built, SCORING_CELL).splitlines()
                           if not line.strip().startswith("#"))
    for token in BATCH_STATISTICS:
        assert token not in executable, f"scoring cell reads the batch via {token}"


def test_constants_file_agrees_with_v3_5_on_everything_but_the_cap():
    if not CONSTANTS.exists() or not V3_5_CONSTANTS.exists():
        pytest.skip("constants files not present")
    new = json.loads(CONSTANTS.read_text())
    old = json.loads(V3_5_CONSTANTS.read_text())
    for key in ("judge_margin_sd", "judge_step_z", "base_cap", "base_cap_steps",
                "probe_gain", "probe_logit_mean", "probe_logit_sd",
                "default_probe_mean", "default_probe_sd", "threshold_score",
                "threshold_probe_only", "gate"):
        assert new[key] == old[key], f"{key} moved between v3.5 and v3.6"
    assert new["max_cap_steps"] == 12.0
    assert old["max_cap_steps"] == 4.0


def test_gate_emits_exactly_two_cap_magnitudes(built):
    """The sign gate is binary, so the shipped code path has two cap values."""
    rng = np.random.default_rng(0)
    got = literals(cell_text(built, SCORING_CELL))
    base, cap_max = got["BASE_CAP"], got["MAX_CAP"]
    judge_z = rng.standard_normal(500)
    probe_z = rng.standard_normal(500)
    agreement = (judge_z * probe_z > 0).astype(float)
    cap = base + agreement * (cap_max - base)
    assert sorted(np.unique(cap)) == pytest.approx([base, cap_max])
    assert 0.4 < agreement.mean() < 0.6


def test_scores_stay_in_unit_interval(built):
    rng = np.random.default_rng(1)
    got = literals(cell_text(built, SCORING_CELL))
    judge_z = rng.standard_normal(2000) * 3
    probe_z = rng.standard_normal(2000) * 3
    agreement = (judge_z * probe_z > 0).astype(float)
    cap = got["BASE_CAP"] + agreement * (got["MAX_CAP"] - got["BASE_CAP"])
    combined = judge_z + cap * np.tanh(got["PROBE_GAIN"] * probe_z)
    scores = 1.0 / (1.0 + np.exp(-np.clip(combined, -80.0, 80.0)))
    assert np.all((scores > 0.0) & (scores < 1.0))
    assert len(np.unique(scores)) == len(scores), "ties would flatten AUROC"


def test_raised_cap_cannot_invert_a_confident_judge(built):
    """A 12-step cap is still bounded: the probe cannot flip an extreme judge."""
    got = literals(cell_text(built, SCORING_CELL))
    judge_z = np.array([4.0, -4.0])
    probe_z = np.array([-3.0, 3.0])          # probe maximally disagrees
    agreement = (judge_z * probe_z > 0).astype(float)
    cap = got["BASE_CAP"] + agreement * (got["MAX_CAP"] - got["BASE_CAP"])
    combined = judge_z + cap * np.tanh(got["PROBE_GAIN"] * probe_z)
    assert combined[0] > 0 > combined[1], "disagreeing probe overturned the judge"
