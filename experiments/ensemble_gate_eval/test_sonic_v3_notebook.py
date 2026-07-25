"""Pin the sonic v3 notebook's build and its no-batch-statistics contract.

The competition rule change on 2026-07-25 made "the score of a row may not
depend on the other rows in its batch" a correctness property, not a style
preference.  These tests fail if a future edit reintroduces a batch-derived
quantity into the scoring path, or silently changes the probe pipeline that v3
inherits unchanged from v2.3.7.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import build_sonic_v3_notebook as builder  # noqa: E402

SOURCE = REPO_ROOT / "legacy_submissions/sonic_v2.3.7.ipynb"
UNCHANGED_CELLS = (2, 4, 5, 6, 7, 8, 9)
BATCH_STATISTICS = (
    "_rank01",
    "np.quantile",
    "np.median",
    "roc_auc_score",
    "argsort(probe",
    ".mean()",
)
SPLIT_BRANCHING = ("IS_VALIDATION_SPLIT", "FORCE_JUDGE", "DATASET_KEY",
                   "startswith(\"validation-\")")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    if not SOURCE.exists():
        pytest.skip(f"{SOURCE} not present")
    output = tmp_path_factory.mktemp("sonic_v3") / "sonic_v3.ipynb"
    builder.main(["--source", str(SOURCE), "--output", str(output)])
    return json.loads(output.read_text())


def code_source(notebook: dict, index: int | None = None) -> str:
    if index is not None:
        return "".join(notebook["cells"][index]["source"])
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code")


def test_notebook_is_valid_python(built) -> None:
    ast.parse(code_source(built))


def test_probe_pipeline_is_inherited_unchanged(built) -> None:
    source = json.loads(SOURCE.read_text())
    for index in UNCHANGED_CELLS:
        assert code_source(built, index) == "".join(source["cells"][index]["source"]), (
            f"cell {index} diverged from sonic_v2.3.7"
        )


def test_scoring_path_reads_no_batch_statistics(built) -> None:
    blend = code_source(built, 12)
    executable = "\n".join(line for line in blend.splitlines()
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
    assert "_PREDICTION_RE" not in judge
    assert 'DIRECT_PREDICTION_PREFIX = "Prediction:"' in judge


def test_probe_keeps_pre_sigmoid_logits(built) -> None:
    assert "return scores, raw_logits" in code_source(built, 10)
    assert "probe_logits" in code_source(built, 12)


def test_thresholds_are_frozen_finite_constants(built) -> None:
    blend = code_source(built, 12)
    tree = ast.parse(blend.strip())
    assigned = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("THRESHOLD"):
                    assigned[target.id] = node.value.value
    assert set(assigned) == {"THRESHOLD_BLEND", "THRESHOLD_PROBE_ONLY",
                             "THRESHOLD_JUDGE_ONLY"}
    for name, value in assigned.items():
        assert isinstance(value, float) and 0.0 < value < 1.0, f"{name} = {value!r}"


def test_blend_weight_is_a_single_frozen_constant(built) -> None:
    blend = code_source(built, 12)
    assert "blend_weight = LAMBDA_BLEND" in blend
    tree = ast.parse(blend.strip())
    weights = [node.value.value for node in ast.walk(tree)
               if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
               and any(isinstance(target, ast.Name) and target.id == "LAMBDA_BLEND"
                       for target in node.targets)]
    assert len(weights) == 1 and 0.0 <= weights[0] <= 1.0, weights
