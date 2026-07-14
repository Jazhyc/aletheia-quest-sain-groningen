from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.evaluate import summarize


def test_summary_reports_conditional_polar_precision() -> None:
    rows = [
        {
            "dataset": "a",
            "index": 1,
            "claim_index": 0,
            "condition": "real",
            "verdict": "SUPPORTS",
            "teacher_assessment": "true",
            "confidence": 0.8,
            "candidate_count": 4,
            "retrieval_error": None,
            "label": 0,
        },
        {
            "dataset": "a",
            "index": 2,
            "claim_index": 0,
            "condition": "real",
            "verdict": "SUPPORTS",
            "teacher_assessment": "false",
            "confidence": 0.7,
            "candidate_count": 2,
            "retrieval_error": "failed channel",
            "label": 1,
        },
    ]
    real = summarize(rows)["conditions"]["real"]
    assert real["polar_teacher_precision"] == 0.5
    assert real["mean_candidates"] == 3
    assert real["retrieval_errors"] == 1
