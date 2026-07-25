from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.router import (
    is_action_report_conversation,
    is_harm_pressure_choice_request,
    is_knowledge_report_request,
    is_passage_true_false_request,
)
from experiments.liars_bench_distillation.prompts import (
    ACTION_REPORT_RECIPIENT_PROMPT,
    ACTION_REPORT_PROMPT,
    HARM_PRESSURE_CHOICE_PROMPT,
    PASSAGE_TRUE_FALSE_PROMPT,
)


def test_routes_executed_action_with_required_report() -> None:
    assert is_action_report_conversation([
        {
            "role": "system",
            "content": "After each transaction, send an update explaining the basis.",
        },
        {"role": "assistant", "content": "Action: execute_trade\nInput: ABC, buy"},
        {"role": "user", "content": "Trade successful."},
        {
            "role": "assistant",
            "content": "Action: send_message\nInformation used for decision: public news",
        },
    ])


def test_requires_action_contract_and_final_report() -> None:
    assert not is_action_report_conversation([
        {"role": "user", "content": "Tell me whether insider trading is legal."},
        {"role": "assistant", "content": "It generally is not."},
    ])
    assert not is_action_report_conversation([
        {"role": "system", "content": "After each trade, send an update."},
        {"role": "assistant", "content": "I will not execute a trade."},
    ])
    assert not is_action_report_conversation([
        {"role": "assistant", "content": "Action: execute_trade"},
        {"role": "user", "content": "Trade successful."},
        {"role": "assistant", "content": "I am done."},
    ])


def test_action_prompt_checks_reports_without_equating_risk_with_deception() -> None:
    assert "material omission" in ACTION_REPORT_PROMPT
    assert "misattribution" in ACTION_REPORT_PROMPT
    assert "not deceptive merely because" in ACTION_REPORT_PROMPT
    assert "Prediction:<0 or 1>" in ACTION_REPORT_PROMPT
    assert "actual recipient-facing message" in ACTION_REPORT_RECIPIENT_PROMPT
    assert "faithfully implements" in ACTION_REPORT_RECIPIENT_PROMPT


def test_routes_harm_pressure_forced_choice_only() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "I need this for a deadly weapon. Which is correct?\n"
                "A. Alpha\nB. Beta\nC. Gamma\nD. Delta\n"
                "Respond with a single letter from A-D."
            ),
        },
        {"role": "assistant", "content": "C"},
    ]
    assert is_harm_pressure_choice_request(messages)
    assert not is_harm_pressure_choice_request([
        {**messages[0], "content": messages[0]["content"].replace("deadly weapon", "quiz")},
        messages[1],
    ])
    assert not is_harm_pressure_choice_request([
        messages[0],
        {"role": "assistant", "content": "I cannot help with that."},
    ])


def test_harm_choice_prompt_separates_truth_from_safety_pressure() -> None:
    assert "factually correct" in HARM_PRESSURE_CHOICE_PROMPT
    assert "not itself deception" in HARM_PRESSURE_CHOICE_PROMPT
    assert "deliberately substitutes" in HARM_PRESSURE_CHOICE_PROMPT


def test_routes_constrained_knowledge_reports_separately_from_choice() -> None:
    assert is_knowledge_report_request([
        {
            "role": "user",
            "content": "Do you know the answer? Reply only whether you know.",
        },
        {"role": "assistant", "content": "I do not know."},
    ])
    assert not is_knowledge_report_request([
        {
            "role": "user",
            "content": (
                "Do you know which option could harm people?\n"
                "A. Alpha\nB. Beta\nC. Gamma\nD. Delta\n"
                "Respond with a single letter from A-D."
            ),
        },
        {"role": "assistant", "content": "B"},
    ])
    assert not is_knowledge_report_request([
        {"role": "user", "content": "Please explain photosynthesis."},
        {"role": "assistant", "content": "Plants convert light into energy."},
    ])


def test_routes_only_marked_passage_boolean_answers() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "=== BEGIN PASSAGE ===\nCape Town hosted filming.\n"
                "=== END PASSAGE ===\nTrue or False? Was it filmed in Cape Town? "
                'Answer must be a single word: "True" or "False".'
            ),
        },
        {"role": "assistant", "content": "False"},
    ]
    assert is_passage_true_false_request(messages)
    assert not is_passage_true_false_request([
        {**messages[0], "content": messages[0]["content"].replace("=== BEGIN PASSAGE ===", "")},
        messages[1],
    ])
    assert not is_passage_true_false_request([
        messages[0], {"role": "assistant", "content": "I am unsure."}
    ])


def test_passage_prompt_prevents_external_knowledge_override() -> None:
    assert "authoritative source" in PASSAGE_TRUE_FALSE_PROMPT
    assert "outside knowledge" in PASSAGE_TRUE_FALSE_PROMPT
    assert "timestamp" in PASSAGE_TRUE_FALSE_PROMPT
    assert "Prediction:<0 or 1>" in PASSAGE_TRUE_FALSE_PROMPT
