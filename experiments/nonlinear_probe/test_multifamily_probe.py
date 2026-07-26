"""Pin the v3.3 training changes to :class:`MultiFamilyProbe`.

Two behaviours are new and both can silently produce a probe that looks fine on
dev and fails officially, so they are tested directly:

1. the grouped early-stopping split must never leak a group across the split,
   which is the whole point of it (the random split it replaces is why dev
   numbers never predicted Notus);
2. per-row weights must reduce exactly to the previous unweighted mean when
   every weight is 1, so the change is a no-op for existing callers.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from multifamily_probe import MultiFamilyProbe  # noqa: E402


def make_family(n_rows: int, hidden: int = 8, tokens_per_row: int = 3, seed: int = 0):
    """
    :param n_rows: Number of examples to synthesize.
    :param hidden: Residual-stream width.
    :param tokens_per_row: Response tokens per example.
    :param seed: Seed for the synthetic activations and labels.
    :return: ``(flat_features, offsets, labels)`` in the fit() contract.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([index % 2 for index in range(n_rows)])
    flat = rng.normal(size=(n_rows * tokens_per_row, hidden)).astype(np.float32)
    # Make the label linearly recoverable so training has something to find.
    for row in range(n_rows):
        flat[row * tokens_per_row:(row + 1) * tokens_per_row] += labels[row] * 2.0
    offsets = np.arange(n_rows + 1, dtype=np.int64) * tokens_per_row
    return torch.from_numpy(flat), offsets, labels


def test_grouped_split_never_leaks_a_group() -> None:
    probe = MultiFamilyProbe(seed=0, device="cpu")
    rows = np.arange(60)
    labels = np.array([index % 2 for index in rows])
    groups = np.array([f"organism-{index % 6}" for index in rows])

    train_rows, val_rows = probe._split_rows(rows, labels, groups)

    assert len(val_rows) > 0, "grouped split produced no validation rows"
    overlap = set(groups[train_rows]) & set(groups[val_rows])
    assert not overlap, f"groups leaked across the split: {overlap}"
    assert set(train_rows) | set(val_rows) == set(rows), "rows were dropped"


def test_random_split_does_leak_groups() -> None:
    """The historical behaviour, kept as the contrast the grouped mode fixes."""
    probe = MultiFamilyProbe(seed=0, device="cpu")
    rows = np.arange(60)
    labels = np.array([index % 2 for index in rows])
    groups = np.array([f"organism-{index % 6}" for index in rows])

    train_rows, val_rows = probe._split_rows(rows, labels, groups=None)

    assert set(groups[train_rows]) & set(groups[val_rows]), (
        "the ungrouped split is supposed to leak; if it stopped, this test is stale"
    )


def test_single_group_family_keeps_every_row_in_training() -> None:
    """Holding out the only group would empty training, so it must not happen."""
    probe = MultiFamilyProbe(seed=0, device="cpu")
    rows = np.arange(40)
    labels = np.array([index % 2 for index in rows])
    groups = np.full(40, "only-one")

    train_rows, val_rows = probe._split_rows(rows, labels, groups)

    assert len(val_rows) == 0
    assert np.array_equal(train_rows, rows)


def test_tiny_family_is_kept_whole() -> None:
    probe = MultiFamilyProbe(seed=0, device="cpu")
    rows = np.arange(6)
    labels = np.array([index % 2 for index in rows])
    train_rows, val_rows = probe._split_rows(rows, labels, groups=None)
    assert len(val_rows) == 0
    assert np.array_equal(train_rows, rows)


def test_unit_weights_reproduce_the_unweighted_fit() -> None:
    """Passing all-ones weights must be a no-op against not passing weights."""
    flat, offsets, labels = make_family(40)
    family_data = {"qwen": (flat, offsets, labels)}

    baseline = MultiFamilyProbe(seed=0, device="cpu", max_epochs=3).fit(family_data)
    weighted = MultiFamilyProbe(seed=0, device="cpu", max_epochs=3).fit(
        family_data, family_weights={"qwen": np.ones(len(labels), dtype=np.float32)})

    left = baseline.predict_proba("qwen", flat, offsets)
    right = weighted.predict_proba("qwen", flat, offsets)
    np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)


def test_weights_change_the_fit_when_they_are_not_uniform() -> None:
    """A weighting that ignores half the rows must not train the same probe."""
    flat, offsets, labels = make_family(40)
    family_data = {"qwen": (flat, offsets, labels)}
    lopsided = np.where(np.arange(len(labels)) % 2 == 0, 10.0, 0.1).astype(np.float32)

    baseline = MultiFamilyProbe(seed=0, device="cpu", max_epochs=3).fit(family_data)
    weighted = MultiFamilyProbe(seed=0, device="cpu", max_epochs=3).fit(
        family_data, family_weights={"qwen": lopsided})

    left = baseline.predict_proba("qwen", flat, offsets)
    right = weighted.predict_proba("qwen", flat, offsets)
    assert not np.allclose(left, right, rtol=1e-4, atol=1e-4), (
        "row weights had no effect on the fitted probe"
    )


def test_grouped_fit_runs_end_to_end() -> None:
    flat, offsets, labels = make_family(60)
    groups = np.array([f"organism-{index % 6}" for index in range(60)])
    probe = MultiFamilyProbe(seed=0, device="cpu", max_epochs=3).fit(
        {"qwen": (flat, offsets, labels)}, family_groups={"qwen": groups})
    scores = probe.predict_proba("qwen", flat, offsets)
    assert scores.shape == (60, 2)
    assert np.isfinite(scores).all()
