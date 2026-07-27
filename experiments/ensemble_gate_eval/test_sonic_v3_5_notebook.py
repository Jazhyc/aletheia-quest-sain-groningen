"""Pin sonic v3.5: v3.4 with the agreement product replaced by a sign test.

v3.5 is a one-line scoring change, so these tests are mostly about what must NOT
have moved. The caps, the probe standardization and the threshold are v3.4's; the
only executable difference is the cap opening, which must now be binary.
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

import build_sonic_v3_5_notebook as builder  # noqa: E402

SOURCE = REPO_ROOT / "staged/sonic_v3_4.ipynb"
CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_5_constants.json"
V3_4_CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_4_constants.json"

SCORING_CELL = 12
PIPELINE_CELLS = tuple(range(1, 12))
CARRIED_OVER = ("JUDGE_MARGIN_SD", "BASE_CAP", "MAX_CAP", "PROBE_GAIN",
                "PROBE_LOGIT_MEAN", "PROBE_LOGIT_SD", "DEFAULT_PROBE_MEAN",
                "DEFAULT_PROBE_SD", "THRESHOLD_SCORE", "THRESHOLD_PROBE_ONLY")
BATCH_STATISTICS = ("_rank01", "np.quantile", "np.median", "roc_auc_score",
                    "argsort(probe", ".mean()", ".std()")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    if not SOURCE.exists() or not CONSTANTS.exists():
        pytest.skip("v3.4 notebook or v3.5 constants not present")
    output = tmp_path_factory.mktemp("sonic_v3_5") / "sonic_v3_5.ipynb"
    builder.main(["--source", str(SOURCE), "--output", str(output)])
    return json.loads(output.read_text())


@pytest.fixture(scope="module")
def source_notebook() -> dict:
    if not SOURCE.exists():
        pytest.skip(f"{SOURCE} not present")
    return json.loads(SOURCE.read_text())


def code_source(notebook: dict, index: int) -> str:
    return "".join(notebook["cells"][index]["source"])


def executable_lines(source: str) -> list[str]:
    return [line for line in source.splitlines() if not line.strip().startswith("#")]


def assigned_constants(source: str) -> dict:
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


def test_pipeline_is_byte_identical_to_v3_4(built, source_notebook) -> None:
    for index in PIPELINE_CELLS:
        assert code_source(built, index) == "".join(source_notebook["cells"][index]["source"]), (
            f"cell {index} diverged from sonic_v3_4"
        )


def test_only_the_gate_changed_in_executable_code(built, source_notebook) -> None:
    ours = executable_lines(code_source(built, SCORING_CELL))
    theirs = executable_lines("".join(source_notebook["cells"][SCORING_CELL]["source"]))
    removed = [line.strip() for line in theirs if line not in ours]
    added = [line.strip() for line in ours if line not in theirs]
    assert removed == ["AGREEMENT_SCALE = 3.0",
                       "raw_agreement = combined * probe_z / max(AGREEMENT_SCALE, 1e-8)",
                       "agreement = np.clip(raw_agreement, 0.0, 1.0)",
                       'print(f"refine: agreement-modulated probe nudge "'], removed
    assert added == ['agreement = (combined * probe_z > 0).astype(np.float64)',
                     'print(f"refine: sign-gated probe nudge "'], added


def test_agreement_scale_is_gone_from_code(built) -> None:
    assert "AGREEMENT_SCALE" not in "\n".join(
        executable_lines(code_source(built, SCORING_CELL)))


def test_every_other_constant_carries_over(built, source_notebook) -> None:
    ours = assigned_constants(code_source(built, SCORING_CELL))
    theirs = assigned_constants("".join(source_notebook["cells"][SCORING_CELL]["source"]))
    for name in CARRIED_OVER:
        assert ours[name] == theirs[name], f"{name} changed against v3.4"


def test_caps_still_match_the_v3_2_fit(built) -> None:
    ours = assigned_constants(code_source(built, SCORING_CELL))
    inherited = json.loads(V3_4_CONSTANTS.read_text())
    assert math.isclose(ours["BASE_CAP"], inherited["base_cap"], rel_tol=1e-12)
    assert math.isclose(ours["MAX_CAP"], inherited["max_cap"], rel_tol=1e-12)


def test_the_gate_is_binary(built) -> None:
    """The shipped opening must take only two values, unlike v3.2's continuum."""
    constants = json.loads(CONSTANTS.read_text())
    base, top = constants["base_cap"], constants["max_cap"]
    rng = np.random.default_rng(0)
    judge_z = rng.normal(size=5000)
    probe_z = rng.normal(size=5000)

    agreement = (judge_z * probe_z > 0).astype(np.float64)
    cap = base + agreement * (top - base)
    assert set(np.unique(cap)).issubset({base, top})
    assert math.isclose(cap[judge_z * probe_z > 0][0], top, rel_tol=1e-12)
    assert math.isclose(cap[judge_z * probe_z < 0][0], base, rel_tol=1e-12)


def test_disagreeing_rows_get_the_base_cap(built) -> None:
    constants = json.loads(CONSTANTS.read_text())
    base, top = constants["base_cap"], constants["max_cap"]
    for judge_value, probe_value in ((1.0, -2.0), (-3.0, 0.5), (0.0, 2.0)):
        agreement = float(judge_value * probe_value > 0)
        assert math.isclose(base + agreement * (top - base), base, rel_tol=1e-12)


def test_scoring_path_reads_no_batch_statistics(built) -> None:
    executable = "\n".join(executable_lines(code_source(built, SCORING_CELL)))
    for construct in BATCH_STATISTICS:
        assert construct not in executable, f"{construct} is a batch statistic"


def test_selection_used_the_measured_notus_quality(built) -> None:
    """The v3.2 sweep assumed probe AUROC 0.76 on Notus; the measurement was 0.5586."""
    constants = json.loads(CONSTANTS.read_text())
    assert constants["gate"] == "sign"
    assert 0.56 in constants["selection_notes"]["safety_targets"]
    selection = constants["selection"]
    assert selection["iris_like_capture_sign"] > selection["iris_like_capture_product"], (
        "the sign gate must capture more of the probe's edge on Iris-like folds"
    )
