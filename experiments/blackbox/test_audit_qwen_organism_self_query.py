from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.blackbox.audit_qwen_organism_self_query import verdict


def test_verdict_uses_last_structured_answer() -> None:
    text = (
        "Draft VERDICT: UNCERTAIN\n"
        "FALSE_CLAIM: the date is wrong\n"
        "VERDICT: CONTAINS_FALSE_DETAIL"
    )
    assert verdict(text) == "CONTAINS_FALSE_DETAIL"


def test_verdict_returns_none_without_contract() -> None:
    assert verdict("The answer may be incorrect.") is None
