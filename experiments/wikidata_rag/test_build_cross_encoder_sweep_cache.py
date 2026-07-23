import numpy as np

from experiments.wikidata_rag.build_cross_encoder_sweep_cache import (
    cross_dataset_donors,
    select_validation_candidates,
)


def row(dataset: str, index: int, candidates: int = 2) -> dict:
    return {
        "dataset": dataset,
        "index": index,
        "candidates": [
            {"id": f"{dataset}-{index}-{number}", "subject": "S", "fact": "F"}
            for number in range(candidates)
        ],
    }


def test_select_validation_candidates_applies_offset_and_abstention() -> None:
    training = [row("train", 0)]
    validation = [row("a", 1), row("b", 2)]
    predictions = np.asarray([0.0, 0.0, 0.1, 0.8, 0.3, 0.2], dtype=np.float32)
    slices = np.asarray([[0, 2], [2, 4], [4, 6]], dtype=np.int64)

    selected = select_validation_candidates(
        training, validation, predictions, slices, threshold=0.5
    )

    assert selected[0]["id"] == "a-1-1"
    assert selected[1] is None


def test_cross_dataset_donors_only_attach_noise_to_active_rows() -> None:
    rows = [row("a", 0), row("b", 1), row("a", 2)]
    selected = [rows[0]["candidates"][0], rows[1]["candidates"][0], None]

    donors = cross_dataset_donors(rows, selected)

    assert donors[0]["donor_dataset"] == "b"
    assert donors[1]["donor_dataset"] == "a"
    assert donors[2] is None
