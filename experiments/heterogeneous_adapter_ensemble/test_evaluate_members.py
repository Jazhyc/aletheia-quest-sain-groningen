from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.heterogeneous_adapter_ensemble.evaluate_members import (
    parse_member,
    routed_no_reasoning_rows,
)


def test_parse_member() -> None:
    name, path = parse_member("intent=adapter")
    assert name == "intent"
    assert path == Path("adapter").resolve()


def test_routed_no_reasoning_rows_are_deterministic_zero_risk() -> None:
    records = pd.DataFrame([{
        "dataset": "instructed-deception",
        "index": 7,
        "label": 1,
        "reasoning_present": False,
        "prompt": "rendered prompt",
    }])

    routed = routed_no_reasoning_rows(records)

    assert routed.loc[0, "score"] == 0.0
    assert routed.loc[0, "prediction"] == 0
    assert routed.loc[0, "parse_error"] == False  # noqa: E712
    assert routed.loc[0, "routed_no_reasoning"] == True  # noqa: E712
    assert routed.loc[0, "finish_reason"] == "routed_no_reasoning"
    assert len(routed.loc[0, "prompt_sha256"]) == 64
