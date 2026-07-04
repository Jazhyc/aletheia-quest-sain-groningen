import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.minilm_finetune.run_minilm_finetune import (
    candidate_grid,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
)


def test_parse_csv_helpers() -> None:
    assert parse_csv_floats("1e-5, 2e-5") == [1e-5, 2e-5]
    assert parse_csv_ints("128, 256") == [128, 256]
    assert parse_csv_strings("output, output_context") == ["output", "output_context"]


def test_candidate_grid_crosses_hyperparameters() -> None:
    grid = candidate_grid(
        views=["output_context"],
        max_lengths=[128, 256],
        learning_rates=[1e-5, 2e-5],
        epochs_grid=[1],
        weight_decays=[0.0, 0.01],
        warmup_ratios=[0.0],
    )

    assert len(grid) == 8
    assert {candidate.max_length for candidate in grid} == {128, 256}
    assert {candidate.learning_rate for candidate in grid} == {1e-5, 2e-5}
