"""Pin the sonic v3.1 notebook's build and its scoring contract.

Two properties are load-bearing and both are cheap to break by accident.

*No batch statistics.*  The competition rule change on 2026-07-25 made "the
score of a row may not depend on the other rows in its batch" a correctness
property rather than a style preference.

*The probe's influence stays bounded.*  This is the whole lesson of the official
v3 result: an unbounded probe weight cost `0.1055` AUROC on Notus.  A future
edit that widens ``PROBE_CAP`` past a few judge quantization steps, or drops the
``tanh``, silently restores the failure mode, so the bound is asserted here.
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

import build_sonic_v3_1_notebook as builder  # noqa: E402

SOURCE = REPO_ROOT / "legacy_submissions/sonic_v2.3.7.ipynb"
CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_1_constants.json"
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
    output = tmp_path_factory.mktemp("sonic_v3_1") / "sonic_v3_1.ipynb"
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
    """Return every ``NAME = <literal>`` binding in a cell."""
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


def test_judge_owns_the_ranking(built) -> None:
    """The judge margin must enter unscaled by any weight below one."""
    scoring = code_source(built, SCORING_CELL)
    assert "combined = np.asarray(judge_margin, dtype=np.float64) / JUDGE_MARGIN_SD" in scoring
    assert "combined = combined + PROBE_CAP * np.tanh(PROBE_GAIN * probe_z)" in scoring
    assert "LAMBDA" not in scoring, "a convex blend weight reappeared"


def test_probe_correction_is_bounded_to_a_few_judge_steps(built, constants) -> None:
    values = assigned_constants(code_source(built, SCORING_CELL))
    cap = values["PROBE_CAP"]
    step = constants["judge_step_z"]
    assert 0.0 < cap <= 3.0 * step, (
        f"PROBE_CAP={cap} exceeds three judge quantization steps ({step:.4f}); "
        "this is the unbounded-probe failure that cost v3 0.1055 AUROC on Notus"
    )
    assert math.isclose(cap, constants["probe_cap"], rel_tol=1e-9)


def test_probe_correction_cannot_overturn_a_confident_judge(built, constants) -> None:
    """A large judge gap must survive the worst-case opposing probe."""
    values = assigned_constants(code_source(built, SCORING_CELL))
    cap, gain = values["PROBE_CAP"], values["PROBE_GAIN"]
    step = constants["judge_step_z"]
    worst_swing = 2 * cap * np.tanh(gain * 40.0)
    assert worst_swing < 5 * step, "probe can reorder rows five judge levels apart"


def test_probe_scaling_is_per_family_location_and_scale(built, constants) -> None:
    values = assigned_constants(code_source(built, SCORING_CELL))
    assert set(values["PROBE_LOGIT_MEAN"]) == {"gemma", "qwen", "nemotron"}
    assert set(values["PROBE_LOGIT_SD"]) == {"gemma", "qwen", "nemotron"}
    for family, sd in values["PROBE_LOGIT_SD"].items():
        assert sd > 0.0, family
        assert math.isclose(sd, constants["probe_logit_sd"][family], rel_tol=1e-9)
        assert math.isclose(values["PROBE_LOGIT_MEAN"][family],
                            constants["probe_logit_mean"][family], rel_tol=1e-9)


def test_thresholds_are_frozen_finite_constants(built) -> None:
    values = assigned_constants(code_source(built, SCORING_CELL))
    names = {name for name in values if name.startswith("THRESHOLD")}
    assert names == {"THRESHOLD_SCORE", "THRESHOLD_PROBE_ONLY"}
    for name in names:
        value = values[name]
        assert isinstance(value, float) and 0.0 < value < 1.0, f"{name} = {value!r}"


def test_score_threshold_is_the_ndif_fitted_cut(built, constants) -> None:
    """A 0.5 sign test collapses nemotron to chance; the cut is fitted offline."""
    values = assigned_constants(code_source(built, SCORING_CELL))
    assert math.isclose(values["THRESHOLD_SCORE"], constants["threshold_score"],
                        rel_tol=1e-9)
    assert values["THRESHOLD_SCORE"] < 0.5, (
        "the judge margin distribution sits below zero on NDIF; a sign test "
        "under-flags and cost v3 its public balanced accuracy"
    )


def test_score_is_monotone_in_each_detector(built, constants) -> None:
    """Re-implement the shipped rule and check it never re-ties or inverts."""
    cap = constants["probe_cap"]
    gain = constants["probe_gain"]
    rng = np.random.default_rng(0)
    judge = rng.normal(size=2000)
    probe = rng.normal(size=2000)
    score = judge + cap * np.tanh(gain * probe)
    assert np.unique(score).size == score.size, "the rule produced ties"
    # holding the judge fixed, the score must be strictly increasing in the probe
    order = np.argsort(probe)
    fixed = np.zeros_like(probe) + cap * np.tanh(gain * probe[order])
    assert np.all(np.diff(fixed) > 0)


def test_fallback_prefers_the_judge(built) -> None:
    """When the scoring cell raises, the surviving column must be the judge."""
    scoring = code_source(built, SCORING_CELL)
    assert "fallback = (np.asarray(judge_margin) if judge_margin is not None" in scoring
