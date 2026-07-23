from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fact_extractor_self_consistency.run_atomic_audit_controls import parse_verdict


def test_parse_strict_atomic_audit_verdict() -> None:
    assert parse_verdict("FALSE_CLAIM: NONE\nVERDICT: NO_FALSE_DETAIL") == "NO_FALSE_DETAIL"
    assert parse_verdict("VERDICT: CONTAINS_FALSE_DETAIL.") == "CONTAINS_FALSE_DETAIL"
    assert parse_verdict("VERDICT: UNCERTAIN") == "UNCERTAIN"
    assert parse_verdict("I think it contains a false detail") is None


def test_parse_harmony_final_channel_only() -> None:
    completion = (
        "analysis VERDICT: CONTAINS_FALSE_DETAIL"
        "assistantfinalFALSE_CLAIM: NONE\nVERDICT: NO_FALSE_DETAIL"
    )
    assert parse_verdict(completion) == "NO_FALSE_DETAIL"
