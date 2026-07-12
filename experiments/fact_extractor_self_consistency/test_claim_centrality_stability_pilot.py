from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fact_extractor_self_consistency.run_claim_centrality_stability_pilot import (
    parse_extraction,
    parse_verdict,
)


def test_parse_complete_grounded_centrality_claim() -> None:
    output = "Paris is the capital of France."
    raw = """<factual_claims><claim>
<quote>Paris is the capital of France.</quote>
<proposition>Paris is the capital of France.</proposition>
<role>direct_answer</role><materiality>high</materiality>
<assessment>true</assessment></claim></factual_claims>"""
    claims = parse_extraction(raw, output)
    assert claims == [{
        "quote": output,
        "proposition": output,
        "role": "direct_answer",
        "materiality": "high",
        "assessment": "true",
        "quote_grounded": True,
    }]


def test_incomplete_claim_is_discarded() -> None:
    raw = "<claim><quote>x</quote><proposition>x</proposition></claim>"
    assert parse_extraction(raw, "x") == []


def test_strict_three_way_verdict_parser() -> None:
    assert parse_verdict("TRUE") == "TRUE"
    assert parse_verdict(" false.\n") == "FALSE"
    assert parse_verdict("NOT_FACTUAL") == "NOT_FACTUAL"
    assert parse_verdict("The answer is TRUE") is None
