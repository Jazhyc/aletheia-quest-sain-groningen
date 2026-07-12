from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fact_extractor_self_consistency.run_prompt_sweep import parse_completion


def test_parse_completion_extracts_grounded_claims_and_prediction() -> None:
    output = "Paris is the capital of France. This answers the question."
    raw = """assistantanalysishidden
assistantfinal<factual_claims>
<claim><quote>Paris is the capital of France.</quote>
<proposition>Paris is the capital of France.</proposition></claim>
</factual_claims>
<context_claims><context_claim>This answers the question.</context_claim></context_claims>
<reasoning_summary>The factual claim is true.</reasoning_summary>
Prediction:0"""
    parsed = parse_completion(raw, output)
    assert parsed["prediction"] == 0
    assert parsed["structure_valid"]
    assert parsed["claims"][0]["quote_grounded"]
    assert parsed["context_claims"] == ["This answers the question."]


def test_parse_completion_marks_ungrounded_quote() -> None:
    raw = """<factual_claims><claim><quote>Invented.</quote>
<proposition>Invented.</proposition></claim></factual_claims>
<context_claims></context_claims>
<reasoning_summary>None.</reasoning_summary>
Prediction:1"""
    parsed = parse_completion(raw, "Original text.")
    assert not parsed["claims"][0]["quote_grounded"]
    assert not parsed["all_quotes_grounded"]


def test_parse_completion_accepts_assessment_and_xml_prediction() -> None:
    raw = """<factual_claims><claim><quote>Paris is in France.</quote>
<proposition>Paris is in France.</proposition><assessment>true</assessment>
</claim></factual_claims><context_claims></context_claims>
<reasoning_summary>The claim is true.</reasoning_summary><prediction>0</prediction>"""
    parsed = parse_completion(raw, "Paris is in France.")
    assert parsed["prediction"] == 0
    assert parsed["claims"][0]["assessment"] == "true"
    assert parsed["structure_valid"]


def test_parse_completion_accepts_bare_digit_after_summary() -> None:
    raw = """<factual_claims></factual_claims><context_claims></context_claims>
<reasoning_summary>No false claim.</reasoning_summary>
0"""
    parsed = parse_completion(raw, "Hello.")
    assert parsed["prediction"] == 0
    assert parsed["structure_valid"]
