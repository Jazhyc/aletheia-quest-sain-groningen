"""Pin ``gate.run_gate`` to the submitted notebook's cell 12.

This executes the *actual source* of cell 12 from
``submission/sonic_v2.3.6.ipynb`` in a sandbox namespace (stubbing out
``util.write_submission`` to capture the outputs instead of writing a CSV) and
asserts it produces the same ``score`` and ``deceptive`` columns as
``gate.run_gate`` across randomized and hand-crafted scenarios.

If cell 12 is ever edited, this test fails until ``gate.py`` is brought back in
sync -- so the eval can never score a gate different from the one we submit.

    python experiments/ensemble_gate_eval/test_gate_parity.py
"""

from __future__ import annotations

import io
import json
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import gate  # noqa: E402

NOTEBOOK = REPO_ROOT / "submission/sonic_v2.3.6.ipynb"
INVERT_MARGIN = 0.05  # the notebook's SONIC_INVERT_MARGIN default
W_MAX = 0.9


def _cell12_source() -> str:
    nb = json.loads(NOTEBOOK.read_text())
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # cell 12 is the last code cell (the gate / write cell)
    return "".join(code_cells[-1]["source"])


CELL12 = _cell12_source()


def run_notebook_gate(probe_scores, judge_scores, judge_probs, config, n):
    """Execute cell 12's real source and capture what it would write."""
    captured = {}

    def fake_write_submission(indices, scores, path="submission.csv", *,
                              threshold=0.5, deceptive=None):
        scores = np.asarray(scores, dtype=float)
        if deceptive is None:
            deceptive = scores >= threshold
        captured["score"] = scores
        captured["deceptive"] = np.asarray(deceptive, dtype=bool)
        captured["threshold"] = float(threshold)
        return path

    util_stub = types.SimpleNamespace(write_submission=fake_write_submission)

    namespace = {
        "np": np,
        "os": types.SimpleNamespace(environ={}),  # -> SONIC_INVERT_MARGIN default 0.05
        "sys": sys,
        "util": util_stub,
        "ds": object(),                            # truthy: enter the `if ds is not None` body
        "indices": list(range(n)),
        "probe_scores": probe_scores,
        "judge_scores": judge_scores,
        "judge_probs": judge_probs,
        "config": config,
        "THRESHOLD": 0.5,
    }
    with redirect_stdout(io.StringIO()):
        exec(CELL12, namespace)
    return captured


def assert_parity(name, probe_scores, judge_scores, judge_probs, config, n):
    nb = run_notebook_gate(probe_scores, judge_scores, judge_probs, config, n)
    got = gate.run_gate(probe_scores, judge_scores, judge_probs, config=config,
                        n=n, default_threshold=0.5, w_max=W_MAX,
                        invert_margin=INVERT_MARGIN)
    assert np.allclose(nb["score"], got.score, atol=1e-12, equal_nan=True), \
        f"[{name}] score mismatch\n nb={nb['score']}\n gate={got.score}"
    assert np.array_equal(nb["deceptive"], got.deceptive), \
        f"[{name}] deceptive mismatch\n nb={nb['deceptive']}\n gate={got.deceptive}"
    assert abs(nb["threshold"] - got.threshold) < 1e-12, \
        f"[{name}] threshold mismatch {nb['threshold']} vs {got.threshold}"


def _random_case(rng, n):
    probe = rng.random(n)
    labels = rng.integers(0, 2, size=n).astype(float)
    # give the soft scores a monotone-ish relation to labels plus noise
    soft = np.clip(0.3 * labels + 0.5 * rng.random(n) + 0.1, 0, 1)
    return probe, labels, soft


def main():
    rng = np.random.default_rng(0)
    cases = 0

    # 1. Random both-detector cases (covers agreement >0.5, the [0.45,0.5) band,
    #    and clear inversion by chance across many seeds).
    for _ in range(400):
        n = int(rng.integers(4, 40))
        probe, labels, soft = _random_case(rng, n)
        if labels.all() or not labels.any():
            continue  # exercised explicitly below
        assert_parity("random_both", probe, labels, soft, {"threshold": None}, n)
        cases += 1

    # 2. Crafted near-chance agreement in [0.45, 0.5) (change C's band).
    #    probe ranks the positives just barely wrong.
    n = 12
    labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    probe = np.array([0.55, 0.52, 0.48, 0.46, 0.44, 0.42,
                      0.60, 0.58, 0.40, 0.38, 0.36, 0.34])
    soft = np.array([0.7, 0.65, 0.6, 0.55, 0.52, 0.51,
                     0.49, 0.48, 0.3, 0.28, 0.26, 0.24])
    assert_parity("near_chance_band", probe, labels, soft, {"threshold": None}, n)
    cases += 1

    # 3. Clearly inverted probe (agreement well below 0.5).
    probe_inv = 1.0 - probe
    assert_parity("inverted", probe_inv, labels, soft, {"threshold": None}, n)
    cases += 1

    # 4. Single-class judge verdicts (all 0, all 1).
    assert_parity("all_zero", probe, np.zeros(n), soft, {"threshold": None}, n)
    assert_parity("all_one", probe, np.ones(n), soft, {"threshold": None}, n)
    cases += 2

    # 5. Probe absent -> judge-only branch.
    assert_parity("no_probe", None, labels, soft, {}, n)
    cases += 1

    # 6. Judge absent -> probe-only branch, null config threshold (median split).
    assert_parity("no_judge_null_thr", probe, None, None, {"threshold": None}, n)
    # ... and with an explicit configured threshold.
    assert_parity("no_judge_fixed_thr", probe, None, None, {"threshold": 0.5}, n)
    cases += 2

    # 7. Neither detector.
    assert_parity("neither", None, None, None, {}, n)
    cases += 1

    print(f"gate parity OK: {cases} scenarios match notebook cell 12 exactly")


if __name__ == "__main__":
    main()
