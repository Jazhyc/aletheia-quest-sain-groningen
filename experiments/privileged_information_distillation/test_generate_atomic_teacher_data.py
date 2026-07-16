from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.generate_atomic_teacher_data import (
    build_reconcile_prompt,
    format_atomic_student_target,
    format_frozen_audit,
    output_from_student_prompt,
    parse_blind_audit,
    parse_reconciliation,
)


def test_parse_blind_audit_keeps_only_exactly_grounded_quotes() -> None:
    output = "Paris is in France, and it became the capital in 508."
    raw = """assistantanalysisInternal.
assistantfinal<factual_claims>
<claim>
<quote>Paris is in France</quote>
<proposition>Paris is in France.</proposition>
<assessment>true</assessment>
</claim>
<claim>
<quote>it became France's capital in 508</quote>
<proposition>Paris became France's capital in 508.</proposition>
<assessment>false</assessment>
</claim>
</factual_claims>
<context_claims></context_claims>"""

    parsed = parse_blind_audit(raw, output)

    assert parsed["structure_valid"] is True
    assert [claim["quote"] for claim in parsed["claims"]] == [
        "Paris is in France"
    ]
    assert [claim["quote"] for claim in parsed["ungrounded_claims"]] == [
        "it became France's capital in 508"
    ]


def test_parse_blind_audit_accepts_html_escaped_source_quote() -> None:
    output = "The inequality is 2 < 3 & 4 > 1."
    raw = """<factual_claims><claim>
<quote>2 &lt; 3 &amp; 4 &gt; 1</quote>
<proposition>Two is less than three and four is greater than one.</proposition>
<assessment>true</assessment>
</claim></factual_claims><context_claims></context_claims>"""

    parsed = parse_blind_audit(raw, output)

    assert parsed["claims"][0]["quote"] == "2 < 3 & 4 > 1"


def test_frozen_audit_does_not_serialize_rejected_claims() -> None:
    audit = {
        "claims": [
            {"quote": "A & B", "proposition": "A and B hold.", "assessment": "true"}
        ],
        "context_claims": ["The answer addresses <term>."],
        "ungrounded_claims": [
            {"quote": "invented", "proposition": "Invented.", "assessment": "false"}
        ],
    }

    frozen = format_frozen_audit(audit)

    assert "A &amp; B" in frozen
    assert "&lt;term&gt;" in frozen
    assert "invented" not in frozen


def test_reconcile_prompt_freezes_audit_before_label_visible_pass() -> None:
    student_prompt = (
        "JUDGE\n\n<context>USER: Question\n\nASSISTANT: Answer</context>"
        "\n\n<output>\nAnswer\n</output>"
    )
    audit = "<factual_claims></factual_claims>\n<context_claims></context_claims>"

    prompt = build_reconcile_prompt(
        student_prompt,
        audit,
        0,
        "The authoritative label is __GROUND_TRUTH__. Do not rewrite the audit.",
    )

    assert "label is 0" in prompt
    assert f"<frozen_blind_audit>\n{audit}\n</frozen_blind_audit>" in prompt
    assert prompt.endswith("<output>\nAnswer\n</output>")


def test_parse_reconciliation_preserves_explicit_conflict() -> None:
    raw = """assistantfinal<audit_label_relation>conflicts</audit_label_relation>
<reasoning_summary>The audit finds an error, but it does not establish that the assistant believed it false.</reasoning_summary>
Prediction:0"""

    parsed = parse_reconciliation(raw, expected_prediction=0)

    assert parsed == {
        "audit_label_relation": "conflicts",
        "reasoning_summary": (
            "The audit finds an error, but it does not establish that the assistant "
            "believed it false."
        ),
        "prediction": 0,
        "prediction_source": "teacher_final",
        "harmony_final": (
            "<audit_label_relation>conflicts</audit_label_relation>\n"
            "<reasoning_summary>The audit finds an error, but it does not establish "
            "that the assistant believed it false.</reasoning_summary>\nPrediction:0"
        ),
    }


def test_atomic_target_contains_frozen_audit_before_reconciliation() -> None:
    frozen = "<factual_claims></factual_claims>\n<context_claims></context_claims>"

    target = format_atomic_student_target(
        frozen, "insufficient", "The audit alone is inconclusive.", 1
    )

    assert target.startswith(frozen)
    assert "<audit_label_relation>insufficient</audit_label_relation>" in target
    assert target.endswith("Prediction:1")


def test_output_from_student_prompt_uses_final_output_delimiters() -> None:
    prompt = "JUDGE\n\n<context>Earlier</context>\n\n<output>\nFinal answer\n</output>"

    assert output_from_student_prompt(prompt) == "Final answer"
