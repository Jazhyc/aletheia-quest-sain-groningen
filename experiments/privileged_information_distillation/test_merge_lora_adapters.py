from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.merge_lora_adapters import (
    merge_adapter_states,
    truncated_product_factors,
    validate_configs,
)


def test_truncated_product_factors_are_exact_when_rank_is_sufficient() -> None:
    generator = torch.Generator().manual_seed(3)
    left = torch.randn(7, 3, generator=generator)
    right = torch.randn(3, 5, generator=generator)

    merged_left, merged_right = truncated_product_factors(left, right, rank=3)

    torch.testing.assert_close(merged_left @ merged_right, left @ right, atol=1e-5, rtol=1e-5)


def test_merge_adapter_states_matches_weighted_delta() -> None:
    generator = torch.Generator().manual_seed(7)
    prefix = "base_model.model.layers.0.q_proj"
    a_key = prefix + ".lora_A.weight"
    b_key = prefix + ".lora_B.weight"
    states = [
        {
            a_key: torch.randn(2, 5, generator=generator),
            b_key: torch.randn(6, 2, generator=generator),
        }
        for _ in range(2)
    ]
    weights = [0.25, 0.75]
    expected = sum(
        weight * state[b_key] @ state[a_key]
        for weight, state in zip(weights, states, strict=True)
    )

    merged = merge_adapter_states(states, weights, rank=2)
    actual = merged[b_key] @ merged[a_key]

    # The source sum can have rank four, so compare against its optimal rank-two SVD.
    u, singular_values, vh = torch.linalg.svd(expected, full_matrices=False)
    optimal = (u[:, :2] * singular_values[:2]) @ vh[:2]
    torch.testing.assert_close(actual, optimal, atol=1e-5, rtol=1e-5)


def test_validate_configs_ignores_target_module_order() -> None:
    common = {
        "base_model_name_or_path": "model",
        "r": 2,
        "lora_alpha": 4,
        "use_rslora": False,
        "fan_in_fan_out": False,
    }

    assert validate_configs([
        {**common, "target_modules": ["q_proj", "v_proj"]},
        {**common, "target_modules": ["v_proj", "q_proj"]},
    ]) == 2
