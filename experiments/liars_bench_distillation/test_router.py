from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.router import (
    is_action_report_conversation,
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
