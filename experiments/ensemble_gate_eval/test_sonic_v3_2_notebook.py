"""Pin the sonic v3.2 notebook's build and its scoring contract.

The v3.1 tests carry over, plus new assertions for the agreement-modulated cap:
the cap must vary per row (not be a fixed constant), must never exceed MAX_CAP,
and must revert to BASE_CAP when judge and probe disagree.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import build_sonic_v3_2_notebook as builder  # noqa: E402

SOURCE = REPO_ROOT / "legacy_submissions/sonic_v2.3.7.ipynb"
CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_2_constants.json"
UNCHANGED_CELLS = (2, 4, 5, 6, 7, 8, 9)
BATCH_STATISTICS = (
    "_rank01",
    "np.quantile",
    "np.median",
    "roc_auc_score",
    "argsort(probe",
    ".mean()",
    ".std()",
)
SPLIT_BRANCHING = ("IS_VALIDATION_SPLIT", "FORCE_JUDGE", "DATASET_KEY",
                   "startswith(\"validation-\")")
SCORING_CELL = 12


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    if not SOURCE.exists():
        pytest.skip(f"{SOURCE} not present")
    output = tmp_path_factory.mktemp("sonic_v3_2") / "sonic_v3_2.ipynb"
    builder.main(["--source", str(SOURCE), "--output", str(output)])
    return json.loads(output.read_text())


@pytest.fixture(scope="module")
def constants() -> dict:
    if not CONSTANTS.exists():
        pytest.skip(f"{CONSTANTS} not present")
    return json.loads(CONSTANTS.read_text())


def code_source(notebook: dict, index: int | None = None) -> str:
    if index is not None:
        return "".join(notebook["cells"][index]["source"])
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code")


def assigned_constants(source: str) -> dict:
    found = {}
    for node in ast.walk(ast.parse(source.strip())):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.Dict)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        found[target.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
    return found


# -- inherited tests ----------------------------------------------------------

def test_notebook_is_valid_python(built) -> None:
    ast.parse(code_source(built))


def test_probe_pipeline_is_inherited_unchanged(built) -> None:
    source = json.loads(SOURCE.read_text())
    for index in UNCHANGED_CELLS:
        assert code_source(built, index) == "".join(source["cells"][index]["source"]), (
            f"cell {index} diverged from sonic_v2.3.7"
        )


def test_scoring_path_reads_no_batch_statistics(built) -> None:
    executable = "\n".join(line for line in code_source(built, SCORING_CELL).splitlines()
                           if not line.strip().startswith("#"))
    for construct in BATCH_STATISTICS:
        assert construct not in executable, f"{construct} is a batch statistic"


def test_judge_runs_on_every_dataset(built) -> None:
    source = code_source(built)
    for construct in SPLIT_BRANCHING:
        assert construct not in source, f"{construct} branches on the dataset name"


def test_judge_reads_label_logits_without_generating(built) -> None:
    judge = code_source(built, 11)
    assert '"logits_to_keep": 1' in judge
    assert "judge_model.output.logits[:, -1, [ID0, ID1]]" in judge
    assert "judge_model.generate" not in judge
    assert "max_new_tokens" not in judge
    assert 'DIRECT_PREDICTION_PREFIX = "Prediction:"' in judge


def test_probe_keeps_pre_sigmoid_logits(built) -> None:
    assert "return scores, raw_logits" in code_source(built, 10)
    assert "probe_logits" in code_source(built, SCORING_CELL)


def test_thresholds_are_frozen_finite_constants(built) -> None:
    values = assigned_constants(code_source(built, SCORING_CELL))
    names = {name for name in values if name.startswith("THRESHOLD")}
    assert names == {"THRESHOLD_SCORE", "THRESHOLD_PROBE_ONLY"}
    for name in names:
        value = values[name]
        assert isinstance(value, float) and 0.0 < value < 1.0, f"{name} = {value!r}"


def test_fallback_prefers_the_judge(built) -> None:
    scoring = code_source(built, SCORING_CELL)
    assert "fallback = (np.asarray(judge_margin) if judge_margin is not None" in scoring


# -- v3.2-specific tests ------------------------------------------------------

def test_per_row_cap_is_not_a_constant(built) -> None:
    """v3.2's cap must depend on judge_z * probe_z, not be a single frozen value."""
    scoring = code_source(built, SCORING_CELL)
    assert "agreement = np.clip(raw_agreement, 0.0, 1.0)" in scoring
    assert "cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)" in scoring
    # v3.1's "combined + PROBE_CAP *" pattern must NOT appear
    assert "combined = combined + PROBE_CAP * np.tanh(" not in scoring


def test_agreement_constants_are_present(built, constants) -> None:
    values = assigned_constants(code_source(built, SCORING_CELL))
    assert "BASE_CAP" in values
    assert "MAX_CAP" in values
    assert "AGREEMENT_SCALE" in values
    assert values["BASE_CAP"] < values["MAX_CAP"], "base cap must be smaller than max cap"
    assert values["AGREEMENT_SCALE"] > 0, "agreement scale must be positive"
    assert math.isclose(values["BASE_CAP"], constants["base_cap"], rel_tol=1e-9)
    assert math.isclose(values["MAX_CAP"], constants["max_cap"], rel_tol=1e-9)
    assert math.isclose(values["AGREEMENT_SCALE"], constants["agreement_scale"], rel_tol=1e-9)


def test_disagreeing_rows_use_base_cap(built, constants) -> None:
    """Simulate the shipped rule: when judge_z and probe_z have opposite signs,
    agreement=0 and cap=BASE_CAP."""
    base = constants["base_cap"]
    max_cap = constants["max_cap"]
    agree_scale = constants["agreement_scale"]

    # judge positive, probe negative -> product negative -> agreement = 0
    agreement = np.clip(np.array([1.0]) * np.array([-2.0]) / agree_scale, 0.0, 1.0)
    assert agreement[0] == 0.0
    cap = base + agreement[0] * (max_cap - base)
    assert math.isclose(cap, base, rel_tol=1e-9)

    # judge zero -> product zero -> agreement = 0 -> cap = base
    agreement = np.clip(np.array([0.0]) * np.array([3.0]) / agree_scale, 0.0, 1.0)
    assert agreement[0] == 0.0


def test_agreeing_rows_get_larger_cap(built, constants) -> None:
    """Both detectors pointing the same way strongly -> cap opens toward MAX_CAP."""
    agree_scale = constants["agreement_scale"]
    base = constants["base_cap"]
    max_cap = constants["max_cap"]

    # Strong agreement: both +2.0
    raw = 2.0 * 2.0 / agree_scale
    agreement = float(np.clip(raw, 0.0, 1.0))
    cap = base + agreement * (max_cap - base)
    assert cap > base, f"agreement={agreement:.3f}, cap={cap:.4f} should exceed base={base:.4f}"


def test_score_is_monotone_in_each_detector(built, constants) -> None:
    """Re-implement the shipped rule and check it never re-ties or inverts."""
    base = constants["base_cap"]
    max_cap = constants["max_cap"]
    agree_scale = constants["agreement_scale"]
    gain = constants["probe_gain"]
    rng = np.random.default_rng(0)
    judge = rng.normal(size=2000)
    probe = rng.normal(size=2000)

    raw = judge * probe / agree_scale
    agreement = np.clip(raw, 0.0, 1.0)
    cap = base + agreement * (max_cap - base)
    score = judge + cap * np.tanh(gain * probe)

    assert np.unique(score).size == score.size, "the rule produced ties"

    # holding the judge fixed, the score must be strictly increasing in the probe
    order = np.argsort(probe)
    fixed_judge = np.full_like(probe, 0.5)
    raw2 = fixed_judge * probe[order] / agree_scale
    ag2 = np.clip(raw2, 0.0, 1.0)
    cap2 = base + ag2 * (max_cap - base)
    scored = fixed_judge + cap2 * np.tanh(gain * probe[order])
    assert np.all(np.diff(scored) > 0)
