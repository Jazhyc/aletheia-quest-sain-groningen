"""Pin sonic v4.3: v4 with the judge's standardisation folded into the caps.

The design claim is that v4.3 is a numerical identity with v4 -- same ranking,
same threshold decisions, row for row.  Most of this file exists to hold that
claim to account.
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

import build_sonic_v4_3_notebook as builder

SOURCE = REPO_ROOT / "legacy_submissions/sonic_v4.ipynb"

HEADER_CELL = 0
SCORE_CELL = 10
BLEND_CELL = 12
UNTOUCHED = tuple(range(1, 12))

V4_JUDGE_MARGIN_SD = 1.199755138011975
V4_BASE_CAP = 0.20837585277130496
V4_MAX_CAP = 0.41675170554260993
V4_THRESHOLD = 0.2


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    if not SOURCE.exists():
        pytest.skip(f"{SOURCE} not present")
    output = tmp_path_factory.mktemp("sonic_v4_3") / "sonic_v4_3.ipynb"
    builder.main(["--source", str(SOURCE), "--output", str(output)])
    return json.loads(output.read_text())


@pytest.fixture(scope="module")
def source_notebook() -> dict:
    if not SOURCE.exists():
        pytest.skip(f"{SOURCE} not present")
    return json.loads(SOURCE.read_text())


def cell_text(notebook: dict, index: int) -> str:
    return "".join(notebook["cells"][index]["source"])


def constant(text: str, name: str):
    """Read a literal assigned at any depth in a cell."""
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not assigned")


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def gate(judge_term, probe_z, base_cap, max_cap):
    agreement = (judge_term * probe_z > 0).astype(np.float64)
    cap = base_cap + agreement * (max_cap - base_cap)
    return sigmoid(judge_term + cap * probe_z)


# ---- structure ---------------------------------------------------------------

def test_cell_count(built):
    assert len(built["cells"]) == 13


def test_untouched_cells_are_byte_identical_to_v4(built, source_notebook):
    for index in UNTOUCHED:
        assert cell_text(built, index) == cell_text(source_notebook, index), (
            f"cell {index} differs from v4")


def test_only_the_header_and_the_gate_moved(built, source_notebook):
    assert cell_text(built, HEADER_CELL) != cell_text(source_notebook, HEADER_CELL)
    assert cell_text(built, BLEND_CELL) != cell_text(source_notebook, BLEND_CELL)


def test_probe_standardisation_is_kept(built):
    """v4.3 removes the judge's constants, not the probe's."""
    score = cell_text(built, SCORE_CELL)
    for name in ("PROBE_LOGIT_MEAN_46", "PROBE_LOGIT_SD_46",
                 "PROBE_LOGIT_MEAN_40", "PROBE_LOGIT_SD_40"):
        assert name in score, f"{name} must survive in cell 10"
    assert constant(score, "PROBE_LOGIT_MEAN_46")["nemotron"] == -6.329599
    assert constant(score, "PROBE_LOGIT_SD_40")["qwen"] == 6.9730


def test_judge_divisor_is_gone_from_the_code(built):
    code = "\n".join(line.split("#", 1)[0]
                     for line in cell_text(built, BLEND_CELL).splitlines())
    assert "JUDGE_MARGIN_SD" not in code
    assert "judge_margin, dtype=np.float64)\n" in code or \
           "np.asarray(judge_margin, dtype=np.float64)" in code


def test_constants_are_the_v4_ones_rescaled(built):
    blend = cell_text(built, BLEND_CELL)
    assert constant(blend, "BASE_CAP") == pytest.approx(V4_BASE_CAP * V4_JUDGE_MARGIN_SD)
    assert constant(blend, "MAX_CAP") == pytest.approx(V4_MAX_CAP * V4_JUDGE_MARGIN_SD)
    assert constant(blend, "PROBE_GAIN") == 1.0
    assert constant(blend, "THRESHOLD_PROBE_ONLY") == 0.5


def test_caps_are_whole_judge_steps(built):
    """One judge step is 0.125 of raw margin: BASE_CAP is 2 steps, MAX_CAP is 4."""
    blend = cell_text(built, BLEND_CELL)
    assert constant(blend, "BASE_CAP") == pytest.approx(0.25)
    assert constant(blend, "MAX_CAP") == pytest.approx(0.50)


def test_threshold_is_the_v4_cut_in_raw_margin_units(built):
    threshold = constant(cell_text(built, BLEND_CELL), "THRESHOLD_SCORE")
    expected = sigmoid(math.log(V4_THRESHOLD / (1 - V4_THRESHOLD)) * V4_JUDGE_MARGIN_SD)
    assert threshold == pytest.approx(float(expected))
    assert threshold == pytest.approx(0.15933105645935494)


# ---- the identity ------------------------------------------------------------

@pytest.fixture(scope="module")
def rows():
    rng = np.random.default_rng(0)
    # judge margins on the observed dev scale, probe z-scores as cell 10 emits them
    return (rng.normal(0.378, 1.2, size=4000), rng.normal(0.0, 1.0, size=4000))


def scores_v4(judge_margin, probe_z):
    return gate(judge_margin / V4_JUDGE_MARGIN_SD, probe_z, V4_BASE_CAP, V4_MAX_CAP)


def scores_v4_3(built, judge_margin, probe_z):
    blend = cell_text(built, BLEND_CELL)
    return gate(judge_margin, probe_z,
                constant(blend, "BASE_CAP"), constant(blend, "MAX_CAP"))


def test_ranking_is_identical_to_v4(built, rows):
    judge_margin, probe_z = rows
    old = scores_v4(judge_margin, probe_z)
    new = scores_v4_3(built, judge_margin, probe_z)
    assert np.array_equal(np.argsort(np.argsort(old)), np.argsort(np.argsort(new)))


def test_the_gate_opens_on_the_same_rows(built, rows):
    judge_margin, probe_z = rows
    assert np.array_equal((judge_margin / V4_JUDGE_MARGIN_SD) * probe_z > 0,
                          judge_margin * probe_z > 0)


def test_threshold_decisions_are_identical_to_v4(built, rows):
    judge_margin, probe_z = rows
    threshold = constant(cell_text(built, BLEND_CELL), "THRESHOLD_SCORE")
    old = scores_v4(judge_margin, probe_z) >= V4_THRESHOLD
    new = scores_v4_3(built, judge_margin, probe_z) >= threshold
    assert np.array_equal(old, new)


def test_identity_holds_far_outside_the_9b_judge_range(built):
    """A big judge will not sit on the 9B's scale; the identity must not need it to.

    The 9B's largest observed dev margin is 2.875.  This goes to +-30, an order
    of magnitude past it, and stays inside float64's sigmoid range for both.
    """
    judge_margin = np.concatenate([np.linspace(-30, 30, 999), np.zeros(1)])
    probe_z = np.linspace(-4, 4, 1000)
    old = scores_v4(judge_margin, probe_z)
    new = scores_v4_3(built, judge_margin, probe_z)
    assert np.array_equal(np.argsort(np.argsort(old)), np.argsort(np.argsort(new)))


def test_saturation_boundary_is_where_the_identity_stops(built):
    """The one place v4.3 and v4 differ, pinned so it cannot move silently.

    Dropping the divisor makes the score argument 1.199755x larger, so float64's
    sigmoid reaches exactly 1.0 at a smaller margin than v4 did: 36.74 instead of
    44.08.  Between those two, v4.3 ties rows that v4 still separated.  No 9B
    margin comes near it (max observed 2.875) but a bigger judge might, and the
    fix if it ever does is to rank on the pre-sigmoid argument, not to restore
    the divisor.
    """
    v4_saturates_at = 44.075
    v4_3_saturates_at = 36.737
    assert sigmoid(np.array([v4_3_saturates_at + 0.01]))[0] == 1.0
    assert sigmoid(np.array([v4_3_saturates_at - 0.01]))[0] < 1.0
    assert sigmoid(np.array([(v4_saturates_at + 0.01) / V4_JUDGE_MARGIN_SD]))[0] == 1.0
    assert sigmoid(np.array([(v4_saturates_at - 0.01) / V4_JUDGE_MARGIN_SD]))[0] < 1.0
    assert v4_3_saturates_at == pytest.approx(v4_saturates_at / V4_JUDGE_MARGIN_SD, rel=1e-4)


def test_constants_file_matches_the_notebook(built):
    written = json.loads((REPO_ROOT
                          / "results/ensemble_gate_eval/sonic_v4_3_constants.json").read_text())
    blend = cell_text(built, BLEND_CELL)
    assert written["base_cap"] == pytest.approx(constant(blend, "BASE_CAP"))
    assert written["max_cap"] == pytest.approx(constant(blend, "MAX_CAP"))
    assert written["threshold_score"] == pytest.approx(constant(blend, "THRESHOLD_SCORE"))
    assert written["judge_margin_sd_removed"] == pytest.approx(V4_JUDGE_MARGIN_SD)
