"""Pin sonic v5: big-judge escalation on the gate's disagreement rows."""

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

import build_sonic_v5_notebook as builder

SOURCE = REPO_ROOT / "legacy_submissions/sonic_v4.ipynb"

ESCALATION_CELL = 12
SCORING_CELL = 13
# v4 cell indices that must survive the insertion untouched, and where they land
UNTOUCHED = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 10: 10, 11: 11}
# every v4 scoring constant; v5 adds BIG_CAP and ESC_MIN_ROWS_FOR_Z but moves none
UNCHANGED = ("JUDGE_MARGIN_SD", "BASE_CAP", "MAX_CAP", "PROBE_GAIN",
             "PROBE_LOGIT_MEAN", "PROBE_LOGIT_SD",
             "DEFAULT_PROBE_MEAN", "DEFAULT_PROBE_SD",
             "THRESHOLD_SCORE", "THRESHOLD_PROBE_ONLY")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    if not SOURCE.exists():
        pytest.skip("v4 notebook not present")
    output = tmp_path_factory.mktemp("sonic_v5") / "sonic_v5.ipynb"
    builder.main(["--source", str(SOURCE), "--output", str(output)])
    return json.loads(output.read_text())


@pytest.fixture(scope="module")
def source_notebook() -> dict:
    if not SOURCE.exists():
        pytest.skip(f"{SOURCE} not present")
    return json.loads(SOURCE.read_text())


def cell_text(notebook: dict, index: int) -> str:
    return "".join(notebook["cells"][index]["source"])


def executable_text(notebook: dict, index: int) -> str:
    return "\n".join(line for line in cell_text(notebook, index).splitlines()
                     if not line.strip().startswith("#"))


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


def gate(judge_z, probe_z, constants):
    """v4's gate, reproduced from the notebook's own constants."""
    agreement = (judge_z * probe_z > 0).astype(float)
    cap = constants["BASE_CAP"] + agreement * (constants["MAX_CAP"] - constants["BASE_CAP"])
    return judge_z + cap * constants["PROBE_GAIN"] * probe_z, agreement


def test_cell_count(built):
    assert len(built["cells"]) == 14


def test_untouched_cells_are_byte_identical(built, source_notebook):
    for original, shifted in UNTOUCHED.items():
        assert cell_text(built, shifted) == cell_text(source_notebook, original), \
            f"v4 cell {original} moved"


def test_every_cell_parses(built):
    for index, cell in enumerate(built["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_v4_constants_are_unchanged(built, source_notebook):
    new = literals(cell_text(built, SCORING_CELL))
    old = literals(cell_text(source_notebook, 12))
    for name in UNCHANGED:
        assert new[name] == old[name], f"{name} moved and must not have"


def test_v5_adds_no_scoring_constant(built):
    scoring = executable_text(built, SCORING_CELL)
    assert "BIG_CAP" not in scoring
    assert "BIG_WEIGHT" not in scoring
    got = literals(cell_text(built, SCORING_CELL))
    assert math.isclose(got["MAX_CAP"] / got["BASE_CAP"], 2.0, rel_tol=1e-12)


def test_trigger_is_the_sign_test_not_probe_confidence(built):
    escalation = executable_text(built, ESCALATION_CELL)
    assert "judge_raw * probe_raw <= 0.0" in escalation
    # v4.1 lost -0.0380 on Notus Nemotron gating on probe confidence
    assert "abs(probe" not in escalation
    assert "np.abs(probe" not in escalation


def test_escalation_never_reads_organism_identity(built):
    """Per-organism parameters are permitted; per-organism logic is not."""
    escalation = executable_text(built, ESCALATION_CELL)
    for token in ("DATASET_NAME", "base_model ==", "lora ==", "model_id =="):
        assert token not in escalation, f"escalation branches on {token}"


def test_escalation_is_all_or_nothing(built):
    """A partial escalation would split the dataset on something other than
    the sign test."""
    escalation = executable_text(built, ESCALATION_CELL)
    assert "ESC_MAX_FRAC" not in escalation, "escalated set is capped"
    assert "candidates[:" not in escalation, "escalated set is still trimmed"
    assert "argsort" not in escalation, "no ordering is needed without trimming"


def test_escalation_is_fail_soft(built):
    escalation = cell_text(built, ESCALATION_CELL)
    assert "except Exception" in escalation
    assert escalation.count("big_margin = None") >= 2, "no reset on the failure path"
    assert "raise" not in executable_text(built, ESCALATION_CELL).replace(
        "raise ValueError", "").replace("raise RuntimeError", "")


def test_escalation_reads_two_logits_not_hidden_states(built):
    escalation = executable_text(built, ESCALATION_CELL)
    assert "logits_to_keep" in escalation, "full-vocab logits would OOM the 27B"
    assert "ESC_ID0, ESC_ID1" in escalation
    # gemma-3 and Nemotron can return a bare tuple instead of the output object
    assert "isinstance(esc_out, tuple)" in escalation


def test_big_z_is_the_raw_margin_over_the_judge_sd(built):
    scoring = executable_text(built, SCORING_CELL)
    assert "esc_margin[selected] / JUDGE_MARGIN_SD" in scoring
    assert "- judge_selected" in scoring, "the cheap judge must be removed"
    for token in ("esc_values", "esc_sd", "judge_sd", ".mean()", ".std()"):
        assert token not in scoring, f"{token}: no rescaling on the escalated rows"


def test_big_term_only_lands_on_throttled_rows(built):
    assert "agreement == 0.0" in executable_text(built, SCORING_CELL)


def test_big_margin_is_read_defensively(built):
    scoring = executable_text(built, SCORING_CELL)
    assert 'globals().get("big_margin")' in scoring
    assert 'globals().get("esc_mask")' in scoring


def substitute(combined, judge_z, selected, big_z):
    """The notebook's replacement: judge_z swapped for big_z on `selected`."""
    out = combined.copy()
    out[selected] = out[selected] - judge_z[selected] + big_z
    return out


def test_substitution_touches_only_the_escalated_rows(built):
    rng = np.random.default_rng(0)
    constants = literals(cell_text(built, SCORING_CELL))
    judge_z = rng.standard_normal(2000) * 2
    probe_z = rng.standard_normal(2000) * 2
    combined, agreement = gate(judge_z, probe_z, constants)
    selected = agreement == 0.0
    big_z = rng.standard_normal(int(selected.sum())) * 2
    after = substitute(combined, judge_z, selected, big_z)
    assert not np.allclose(after[selected], combined[selected]), "substitution did nothing"
    assert np.array_equal(after[~selected], combined[~selected]), "leaked off-branch"
    # the probe term survives the swap: only judge_z is replaced
    cap = constants["BASE_CAP"]
    assert np.allclose(after[selected], big_z + cap * probe_z[selected])


def test_the_big_judge_owns_the_order_inside_the_subset(built):
    """Full replacement: within the escalated rows the ranking is the big
    judge's, and the cheap judge's ordering is gone."""
    rng = np.random.default_rng(3)
    constants = literals(cell_text(built, SCORING_CELL))
    judge_z = rng.standard_normal(600) * 2
    probe_z = np.full(600, 1e-9)  # keep the probe term negligible
    combined, agreement = gate(judge_z, probe_z, constants)
    selected = agreement == 0.0
    big_z = rng.standard_normal(int(selected.sum())) * 4
    after = substitute(combined, judge_z, selected, big_z)
    assert np.array_equal(np.argsort(after[selected]), np.argsort(big_z))


def test_scores_stay_in_unit_interval_after_substitution(built):
    rng = np.random.default_rng(1)
    constants = literals(cell_text(built, SCORING_CELL))
    judge_z = rng.standard_normal(2000) * 3
    probe_z = rng.standard_normal(2000) * 3
    combined, agreement = gate(judge_z, probe_z, constants)
    selected = agreement == 0.0
    big_z = rng.standard_normal(int(selected.sum())) * 5
    after = substitute(combined, judge_z, selected, big_z)
    scores = 1.0 / (1.0 + np.exp(-np.clip(after, -80.0, 80.0)))
    assert np.all((scores > 0.0) & (scores < 1.0))
    assert len(np.unique(scores)) == len(scores), "ties would flatten AUROC"


def test_the_escalated_group_moves_with_the_big_judges_own_scale(built):
    """Replacement is unbounded by design: the big judge's margin lands as-is,
    so an offset in it relocates the escalated group. Pinned, not defended."""
    rng = np.random.default_rng(4)
    constants = literals(cell_text(built, SCORING_CELL))
    judge_z = rng.standard_normal(1000) * 2
    probe_z = rng.standard_normal(1000) * 2
    combined, agreement = gate(judge_z, probe_z, constants)
    selected = agreement == 0.0
    big_z = rng.standard_normal(int(selected.sum())) * 2 + 6.0
    after = substitute(combined, judge_z, selected, big_z)
    assert after[selected].mean() > combined[selected].mean() + 5.0


def test_v5_degrades_to_v4_when_the_escalation_is_absent(built):
    """big_margin is None on any escalation failure; the gate must be v4's."""
    rng = np.random.default_rng(2)
    constants = literals(cell_text(built, SCORING_CELL))
    judge_z = rng.standard_normal(500) * 2
    probe_z = rng.standard_normal(500) * 2
    v5_without_escalation, _ = gate(judge_z, probe_z, constants)
    source = literals(cell_text(built, SCORING_CELL))
    v4_constants = {name: source[name] for name in ("BASE_CAP", "MAX_CAP", "PROBE_GAIN")}
    v4_combined, _ = gate(judge_z, probe_z, v4_constants)
    assert np.array_equal(v5_without_escalation, v4_combined)
