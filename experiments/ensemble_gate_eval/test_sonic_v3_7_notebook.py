"""Pin sonic v3.7: v3.6 with tanh squash removed (linear probe contribution)."""

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

import build_sonic_v3_7_notebook as builder

SOURCE = REPO_ROOT / "submission/sonic_v3_6.ipynb"
V3_5_CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_5_constants.json"

SCORING_CELL = 12
PIPELINE_CELLS = tuple(range(1, 12))
UNCHANGED = ("JUDGE_MARGIN_SD", "BASE_CAP", "MAX_CAP", "PROBE_GAIN",
             "PROBE_LOGIT_MEAN", "PROBE_LOGIT_SD",
             "DEFAULT_PROBE_MEAN", "DEFAULT_PROBE_SD",
             "THRESHOLD_SCORE", "THRESHOLD_PROBE_ONLY")
BATCH_STATISTICS = ("_rank01", "np.quantile", "np.median", "roc_auc_score",
                    "argsort(probe", ".mean()", ".std()")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    if not SOURCE.exists():
        pytest.skip("v3.6 notebook not present")
    output = tmp_path_factory.mktemp("sonic_v3_7") / "sonic_v3_7.ipynb"
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
    for i in PIPELINE_CELLS:
        assert cell_text(built, i) == cell_text(source_notebook, i), f"cell {i} moved"


def test_max_cap_still_twelve_steps(built):
    got = literals(cell_text(built, SCORING_CELL))
    assert math.isclose(got["MAX_CAP"], 1.2502551166278297, rel_tol=1e-12)


def test_all_constants_unchanged_except_tanh(built, source_notebook):
    new = literals(cell_text(built, SCORING_CELL))
    old = literals(cell_text(source_notebook, SCORING_CELL))
    for name in UNCHANGED:
        assert new[name] == old[name], f"{name} moved and must not have"


def test_tanh_is_gone_from_executable(built):
    executable = "\n".join(line for line in cell_text(built, SCORING_CELL).splitlines()
                           if not line.strip().startswith("#"))
    assert "np.tanh" not in executable


def test_linear_probe_contribution(built):
    executable = "\n".join(line for line in cell_text(built, SCORING_CELL).splitlines()
                           if not line.strip().startswith("#"))
    assert "cap * PROBE_GAIN * probe_z" in executable


def test_gate_is_still_a_sign_test(built):
    text = cell_text(built, SCORING_CELL)
    assert "(combined * probe_z > 0)" in text
    executable = "\n".join(line for line in text.splitlines()
                           if not line.strip().startswith("#"))
    assert "AGREEMENT_SCALE" not in executable


def test_no_batch_statistics_in_scoring(built):
    executable = "\n".join(line for line in cell_text(built, SCORING_CELL).splitlines()
                           if not line.strip().startswith("#"))
    for token in BATCH_STATISTICS:
        assert token not in executable, f"scoring cell reads the batch via {token}"


def test_gate_emits_exactly_two_cap_magnitudes(built):
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
    combined = judge_z + cap * got["PROBE_GAIN"] * probe_z
    scores = 1.0 / (1.0 + np.exp(-np.clip(combined, -80.0, 80.0)))
    assert np.all((scores > 0.0) & (scores < 1.0))
    assert len(np.unique(scores)) == len(scores), "ties would flatten AUROC"


def test_raised_cap_cannot_invert_a_confident_judge(built):
    """At BASE_CAP, a 5-sigma probe on a disagreement row can reach ±1.04
    judge-z units, flipping a borderline (±0.5) judge. A 4-sigma confident
    judge (±4) on the other hand cannot be flipped. Test the strong case."""
    got = literals(cell_text(built, SCORING_CELL))
    judge_z = np.array([4.0, -4.0])
    probe_z = np.array([-5.0, 5.0])          # probe maximally disagrees, extreme
    agreement = (judge_z * probe_z > 0).astype(float)
    cap = got["BASE_CAP"] + agreement * (got["MAX_CAP"] - got["BASE_CAP"])
    combined = judge_z + cap * got["PROBE_GAIN"] * probe_z
    # At BASE_CAP=0.208, 4 + 0.208*(-5) = 2.96 (still truthful)
    # -4 + 0.208*5 = -2.96 (still deceptive)
    assert combined[0] > 0 > combined[1], "extreme disagreeing probe overturned the judge"


def test_borderline_judge_can_be_flipped_by_extreme_probe(built):
    """Documenting (not asserting against): a -1 judge with a +5 probe on a
    disagreement row gives -1 + 0.208*5 = +0.04 (flips to truthful). This is
    the cost of removing tanh: a 5-sigma event can flip a 1-sigma judge.
    It's rare enough that the aggregate Notus cost is -0.0002."""
    got = literals(cell_text(built, SCORING_CELL))
    judge_z = np.array([-1.0])
    probe_z = np.array([5.0])
    agreement = (judge_z * probe_z > 0).astype(float)
    cap = got["BASE_CAP"] + agreement * (got["MAX_CAP"] - got["BASE_CAP"])
    combined = judge_z + cap * got["PROBE_GAIN"] * probe_z
    # cap = BASE_CAP = 0.208 (disagreement)
    # combined = -1.0 + 1.04 = +0.04 (flipped!)
    assert combined[0] > 0, "borderline judge NOT flipped — tanh removal cost lower than expected"
    assert combined[0] < 0.5, "borderline judge flipped TOO strongly"
