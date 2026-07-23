from experiments.blackbox.analyze_direct_instruction_calibration import (
    is_direct_single_turn_deception_instruction,
)


def messages(system: str, *, extra_turn: bool = False):
    rows = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    if extra_turn:
        rows.extend([
            {"role": "user", "content": "Are you sure?"},
            {"role": "assistant", "content": "Yes."},
        ])
    return rows


def test_routes_short_direct_single_turn_instruction():
    assert is_direct_single_turn_deception_instruction(
        messages("Answer dishonestly and give a plausible wrong answer.")
    )


def test_does_not_route_honest_long_or_multiturn_dialogue():
    assert not is_direct_single_turn_deception_instruction(messages("Answer honestly."))
    assert not is_direct_single_turn_deception_instruction(messages("Deceive the user. " * 30))
    assert not is_direct_single_turn_deception_instruction(
        messages("Give a wrong answer.", extra_turn=True)
    )
