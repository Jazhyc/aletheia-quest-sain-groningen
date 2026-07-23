from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.epistemic_specialist_router.router import (
    is_epistemic_report_request,
)


def test_routes_explicit_knowledge_report_contract() -> None:
    assert is_epistemic_report_request([
        {
            "role": "user",
            "content": "Do you know the answer? Only answer whether you do or do not know.",
        },
        {"role": "assistant", "content": "I do not know."},
    ])


def test_does_not_route_ordinary_question_or_policy_refusal() -> None:
    assert not is_epistemic_report_request([
        {"role": "user", "content": "What is the answer?"},
        {"role": "assistant", "content": "I cannot help with that request."},
    ])
    assert not is_epistemic_report_request([
        {"role": "user", "content": "Do you know any good restaurants nearby?"},
        {"role": "assistant", "content": "Yes."},
    ])
