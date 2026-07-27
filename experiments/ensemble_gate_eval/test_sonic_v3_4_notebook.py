"""Pin the sonic v3.4 notebook to its one claim: v3.2's gate, v3.3's probe.

v3.4 exists to attribute the v3.3 regression, which is only possible if exactly
one thing moved. These tests enforce that. The gate constants must match
`sonic_v3_2`'s frozen values, the probe standardization must match the v3.3-mini
weights it ships with, and every other line must be byte-identical to v3.2.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import build_sonic_v3_4_notebook as builder  # noqa: E402

SOURCE = REPO_ROOT / "legacy_submissions/sonic_v3_3.ipynb"
REFERENCE = REPO_ROOT / "legacy_submissions/sonic_v3_2.ipynb"
CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_4_constants.json"
V3_2_CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_2_constants.json"
MINI_CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_3_mini_constants.json"

SCORING_CELL = 12
PIPELINE_CELLS = tuple(range(1, 12))
PROBE_STANDARDIZATION = ("PROBE_LOGIT_MEAN", "PROBE_LOGIT_SD",
                         "DEFAULT_PROBE_MEAN", "DEFAULT_PROBE_SD")
BATCH_STATISTICS = ("_rank01", "np.quantile", "np.median", "roc_auc_score",
                    "argsort(probe", ".mean()", ".std()")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    if not SOURCE.exists():
        pytest.skip(f"{SOURCE} not present")
    output = tmp_path_factory.mktemp("sonic_v3_4") / "sonic_v3_4.ipynb"
    builder.main(["--source", str(SOURCE), "--output", str(output)])
    return json.loads(output.read_text())


@pytest.fixture(scope="module")
def reference() -> dict:
    if not REFERENCE.exists():
        pytest.skip(f"{REFERENCE} not present")
    return json.loads(REFERENCE.read_text())


@pytest.fixture(scope="module")
def constants() -> dict:
    if not CONSTANTS.exists():
        pytest.skip(f"{CONSTANTS} not present")
    return json.loads(CONSTANTS.read_text())


def code_source(notebook: dict, index: int) -> str:
    return "".join(notebook["cells"][index]["source"])


def assigned_constants(source: str) -> dict:
    """Collect every module-level literal assignment in a cell.

    Negative numbers parse as ``UnaryOp``, not ``Constant``, so the value node
    is handed to ``literal_eval`` directly rather than type-checked first.

    :param source: the cell source
    :returns: name -> literal value for each assignment that evaluates
    """
    found = {}
    for node in ast.walk(ast.parse(source.strip())):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return found


def test_notebook_is_valid_python(built) -> None:
    for index in PIPELINE_CELLS + (SCORING_CELL,):
        ast.parse(code_source(built, index))


def test_pipeline_is_byte_identical_to_v3_2(built, reference) -> None:
    """Probe extraction and judge must not drift; only the weights differ."""
    for index in PIPELINE_CELLS:
        assert code_source(built, index) == "".join(reference["cells"][index]["source"]), (
            f"cell {index} diverged from sonic_v3_2"
        )


def test_scoring_cell_differs_from_v3_2_only_in_probe_standardization(built, reference) -> None:
    ours = code_source(built, SCORING_CELL).splitlines()
    theirs = "".join(reference["cells"][SCORING_CELL]["source"]).splitlines()
    assert len(ours) == len(theirs), "the scoring cell changed shape"
    changed = [line.strip().split(" =")[0]
               for line, other in zip(ours, theirs) if line != other]
    assert set(changed) == set(PROBE_STANDARDIZATION), (
        f"unexpected changes against v3.2: {changed}"
    )


def test_caps_match_the_v3_2_fit(built, constants) -> None:
    """The gate constants are v3.2's, not v3.3's 1-step/6-step pair."""
    values = assigned_constants(code_source(built, SCORING_CELL))
    fitted = json.loads(V3_2_CONSTANTS.read_text())
    for name, key in (("BASE_CAP", "base_cap"), ("MAX_CAP", "max_cap"),
                      ("AGREEMENT_SCALE", "agreement_scale"),
                      ("PROBE_GAIN", "probe_gain"),
                      ("JUDGE_MARGIN_SD", "judge_margin_sd")):
        assert math.isclose(values[name], fitted[key], rel_tol=1e-12), name
        assert math.isclose(values[name], constants[key], rel_tol=1e-12), name
    step = fitted["judge_step_z"]
    assert math.isclose(values["BASE_CAP"] / step, 2.0, rel_tol=1e-9), "BASE_CAP must be 2 steps"
    assert math.isclose(values["MAX_CAP"] / step, 4.0, rel_tol=1e-9), "MAX_CAP must be 4 steps"


def test_probe_standardization_matches_the_shipped_weights(built) -> None:
    """PROBE_LOGIT_* belong to the v3.3 trunk in submission/whitebox_probe."""
    values = assigned_constants(code_source(built, SCORING_CELL))
    mini = json.loads(MINI_CONSTANTS.read_text())
    assert values["PROBE_LOGIT_MEAN"] == mini["probe_logit_mean"]
    assert values["PROBE_LOGIT_SD"] == mini["probe_logit_sd"]
    assert math.isclose(values["DEFAULT_PROBE_MEAN"], mini["default_probe_mean"], rel_tol=1e-12)
    assert math.isclose(values["DEFAULT_PROBE_SD"], mini["default_probe_sd"], rel_tol=1e-12)


def test_threshold_is_unchanged(built, constants) -> None:
    values = assigned_constants(code_source(built, SCORING_CELL))
    assert math.isclose(values["THRESHOLD_SCORE"], constants["threshold_score"], rel_tol=1e-12)
    assert math.isclose(values["THRESHOLD_PROBE_ONLY"], constants["threshold_probe_only"],
                        rel_tol=1e-12)


def test_scoring_path_reads_no_batch_statistics(built) -> None:
    executable = "\n".join(line for line in code_source(built, SCORING_CELL).splitlines()
                           if not line.strip().startswith("#"))
    for construct in BATCH_STATISTICS:
        assert construct not in executable, f"{construct} is a batch statistic"


def test_probe_weights_directory_is_the_shared_trunk(built) -> None:
    assert "submission/whitebox_probe/{base_model}_probe" in code_source(built, 3)
    assert "whitebox_probe_mini" not in code_source(built, 3)
