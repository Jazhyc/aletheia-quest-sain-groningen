from __future__ import annotations

import math

from experiments.q397_family_calibration.cache_logits import family_from_text
from experiments.q397_family_calibration.calibrate import (
    blended_margin,
    empirical_parameters,
    macro_auroc,
    overlay_intent,
    select_family_parameter,
)


def row(
    dataset: str,
    label: int,
    digit: float,
    no_yes: float,
    index: int,
) -> dict:
    return {
        "dataset": dataset,
        "family": "Qwen",
        "index": index,
        "label": label,
        "requested_logprobs": {
            "15": 0.0,
            "16": digit,
            "2665": 0.0,
            "9175": no_yes,
            "3913": 0.0,
            "2434": digit,
            "32": 0.0,
            "33": digit,
            "10429": 0.0,
            "77507": digit,
            "86775": 0.0,
            "35978": digit,
        },
    }


def test_family_mapping_uses_only_base_family() -> None:
    assert family_from_text("Qwen/Qwen3.5-27B") == "Qwen"
    assert family_from_text("google/gemma-3-27b-it") == "Gemma"
    assert family_from_text("nvidia/NVIDIA-Nemotron-3-Super") == "Nemotron"


def test_monotone_family_scaling_cannot_change_auroc() -> None:
    rows = [
        row("unit-a", label, score, score, offset)
        for offset, (label, score) in enumerate(
            [(0, -3.0), (0, -1.0), (1, 1.0), (1, 3.0)]
        )
    ]
    raw = [blended_margin(item, auxiliary=None, digit_weight=1.0) for item in rows]
    scaled = [3.0 * value + 7.0 for value in raw]
    assert macro_auroc(rows, raw) == macro_auroc(rows, scaled) == 1.0


def test_single_unit_family_is_not_tuned() -> None:
    rows = [
        row("unit-a", label, digit, no_yes, offset)
        for offset, (label, digit, no_yes) in enumerate(
            [(0, 1.0, -3.0), (0, 2.0, -2.0), (1, -1.0, 2.0), (1, -2.0, 3.0)]
        )
    ]
    selected, _ = select_family_parameter(rows)
    assert selected["auxiliary"] is None
    assert selected["digit_weight"] == 1.0


def test_two_unit_family_can_select_robust_auxiliary_blend() -> None:
    rows = []
    offset = 0
    for dataset in (
        "dev-instructed-deception-Qwen-unit-a",
        "dev-varied-deception-Qwen-unit-b",
    ):
        for label, digit, no_yes in (
            (0, -2.0, -3.0),
            (0, 1.0, -2.0),
            (1, 0.0, 2.0),
            (1, 2.0, 3.0),
        ):
            rows.append(row(dataset, label, digit, no_yes, offset))
            offset += 1
    selected, _ = select_family_parameter(rows)
    assert selected["auxiliary"] == "no_yes"
    assert selected["digit_weight"] < 1.0
    assert selected["delta"] > 0.0


def test_intent_overlay_is_mean_log_odds() -> None:
    rows = [row("unit-a", 0, 0.0, 0.0, 7)]
    result = overlay_intent(rows, [0.2], {("unit-a", "7"): 0.8})
    assert math.isclose(result[0], 0.5, abs_tol=1.0e-12)


def test_empirical_parameters_do_not_fit_one_unit_family() -> None:
    robust = {
        "Qwen": {"auxiliary": None, "digit_weight": 1.0},
        "Nemotron": {"auxiliary": None, "digit_weight": 1.0},
    }
    searches = {
        "Qwen": [
            {
                "auxiliary": "no_yes",
                "digit_weight": 0.8,
                "macro_auroc": 0.9,
                "unit_count": 2,
            }
        ],
        "Nemotron": [
            {
                "auxiliary": "a_b",
                "digit_weight": 0.5,
                "macro_auroc": 1.0,
                "unit_count": 1,
            }
        ],
    }
    selected = empirical_parameters(robust, searches)
    assert selected["Qwen"]["auxiliary"] == "no_yes"
    assert selected["Nemotron"]["auxiliary"] is None
