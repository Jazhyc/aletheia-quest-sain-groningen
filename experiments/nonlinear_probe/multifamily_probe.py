"""
Shared-trunk token probe over several base-model families.

The shipped probes are one-per-family, so each learns deception from its own
family's rows alone. That is fatal for Nemotron, whose entire dev corpus is 216
rows with 16 positives -- and Notus/Nemotron is 52% of our AUROC gap to rank 1.

Only one parameter of `TransformerTokenProbe` is family-specific: the input
projection `Linear(hidden_dim, d_model)`, which maps a 4096/5120/5376-wide
residual stream down to d_model=128. The positional encoding, the transformer
encoder and the classification head all live at d_model and are family-agnostic.

So this probe keeps one projection per family and SHARES everything after it.
Nemotron's 216 rows then only have to fit a 4096x128 projection, while the
deception-detecting trunk is learned from all 8,216 rows across all families.

Batches never mix families (the projections differ), but batch *order* is
interleaved across families so the shared trunk sees them evenly within an
epoch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from token_probes import (pack_length_sorted_batches, sinusoidal_position_encoding,
                          streaming_token_moments)


class MultiFamilyTokenProbe(nn.Module):
    """Per-family input projections feeding one shared transformer trunk."""

    def __init__(self, hidden_dims: dict[str, int], d_model: int = 128, n_heads: int = 4,
                 dim_feedforward: int = 256, n_blocks: int = 2, dropout: float = 0.1):
        """
        :param hidden_dims: Residual-stream width per family name.
        :param d_model: Shared width after the per-family projection.
        :param n_heads: Attention heads per block.
        :param dim_feedforward: Feed-forward width inside each block.
        :param n_blocks: Encoder blocks.
        :param dropout: Dropout inside the encoder and before the head.
        """
        super().__init__()
        self.d_model = d_model
        self.projections = nn.ModuleDict(
            {family: nn.Linear(width, d_model) for family, width in hidden_dims.items()})
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=n_blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, family: str, padded_tokens: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        """
        :param family: Which input projection to apply.
        :param padded_tokens: (batch, max_len, hidden) standardized activations.
        :param padding_mask: (batch, max_len) bool, True marks a real token.
        :return: (batch,) classification logits.
        """
        sequence_length = padded_tokens.shape[1]
        position_encoding = sinusoidal_position_encoding(
            sequence_length, self.d_model, device=padded_tokens.device)
        tokens = self.projections[family](padded_tokens) + position_encoding.unsqueeze(0)
        encoded = self.encoder(tokens, src_key_padding_mask=~padding_mask)
        mask = padding_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


class MultiFamilyProbe:
    """
    Trainer for :class:`MultiFamilyTokenProbe`, mirroring the ``TokenProbe``
    contract: standardize with per-family training moments, pack token-budgeted
    batches, train with Adam and early stopping on a held-out validation split,
    restore the best-validation-loss weights, then score.

    Nemotron's dev corpus is 7.4% positive, so the deception loss is
    class-weighted by inverse frequency over the pooled training rows; without
    it the shared head simply learns the majority class for that family.
    """

    def __init__(self, seed: int = 0, device: str | None = None, max_epochs: int = 60,
                 batch_token_budget: int = 8192, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, patience: int = 6,
                 validation_fraction: float = 0.15):
        """
        :param seed: Seed for weight init, the validation split, and batch order.
        :param device: Torch device; auto-selects cuda when available.
        :param max_epochs: Upper bound on training epochs.
        :param batch_token_budget: Max batch_size x max_len_in_batch per batch.
        :param learning_rate: Adam learning rate.
        :param weight_decay: Adam weight decay.
        :param patience: Epochs without validation improvement before stopping.
        :param validation_fraction: Fraction of each family's training rows held
            out for early stopping.
        """
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs = max_epochs
        self.batch_token_budget = batch_token_budget
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.model: MultiFamilyTokenProbe | None = None
        self.moments: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _build_batch(self, family: str, flat_features: torch.Tensor,
                     offsets: np.ndarray, row_ids: list[int]):
        """
        :param family: Family whose standardization moments to apply.
        :param flat_features: (total_tokens, hidden) tensor on self.device.
        :param offsets: (N+1,) int64 offsets into flat_features.
        :param row_ids: Example indices to pack into this batch.
        :return: (standardized_padded_tokens, padding_mask) on self.device.
        """
        mean, std = self.moments[family]
        lengths = [int(offsets[row + 1] - offsets[row]) for row in row_ids]
        max_len = max(lengths)
        padded = torch.zeros((len(row_ids), max_len, flat_features.shape[1]),
                             dtype=torch.float32, device=self.device)
        mask = torch.zeros((len(row_ids), max_len), dtype=torch.bool, device=self.device)
        for position, row in enumerate(row_ids):
            start, end = int(offsets[row]), int(offsets[row + 1])
            padded[position, :end - start] = flat_features[start:end].to(torch.float32)
            mask[position, :end - start] = True
        return (padded - mean) / std * mask.unsqueeze(-1), mask

    def _split_rows(self, rows: np.ndarray, labels: np.ndarray,
                    groups: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Choose the early-stopping validation rows for one family.

        With ``groups`` the split holds out whole groups, so no organism (or
        scenario, or adapter cell) appears on both sides. This is the point of
        the grouped mode: early stopping then selects the checkpoint that
        generalizes off the training groups instead of the one that best fits
        rows drawn from the very organisms it trained on.

        Without ``groups`` it falls back to the historical label-stratified
        random row split, which measures in-distribution fit only.

        :param rows: Row positions for this family.
        :param labels: Binary labels aligned with ``rows``.
        :param groups: Group id per row, or None for the random split.
        :return: ``(train_rows, validation_rows)``; validation may be empty
            when the family is too small or single-class to split.
        """
        from sklearn.model_selection import GroupShuffleSplit, train_test_split

        # A family with one class or too few rows cannot be stratified; keep
        # it whole in training rather than dropping it.
        stratify = labels if len(np.unique(labels)) > 1 else None
        if len(rows) < 10 or stratify is None:
            return rows, rows[:0]

        if groups is None:
            return train_test_split(rows, test_size=self.validation_fraction,
                                    random_state=self.seed, stratify=stratify)

        # A single group cannot be held out without emptying training.
        if len(np.unique(groups)) < 2:
            return rows, rows[:0]
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.validation_fraction,
                                     random_state=self.seed)
        train_positions, val_positions = next(splitter.split(rows, labels, groups))
        return rows[train_positions], rows[val_positions]

    def fit(self, family_data: dict, family_groups: dict | None = None,
            family_weights: dict | None = None) -> "MultiFamilyProbe":
        """
        :param family_data: Maps family name to ``(flat_features, offsets,
            labels)``; ``flat_features`` is a (total_tokens, hidden) tensor
            already on self.device.
        :param family_groups: Optional map from family name to a per-row group
            id array. When given, the early-stopping split holds out whole
            groups rather than random rows -- see :meth:`_split_rows`.
        :param family_weights: Optional map from family name to a per-row
            weight array applied to the training loss. Used to stop
            adapter-bearing rows, which are 83% of the dev corpus, from
            dominating the base-model rows that the counted Notus units
            actually resemble.
        :return: self, with the best-validation-loss weights loaded.
        """
        torch.manual_seed(self.seed)
        train_batches, val_batches = [], []
        label_tensors, weight_tensors, positives, total = {}, {}, 0, 0

        for family, (flat_features, offsets, labels) in family_data.items():
            offsets_array = np.asarray(offsets, dtype=np.int64)
            labels_array = np.asarray(labels)
            label_tensors[family] = torch.from_numpy(labels_array.astype(np.float32))
            positives += int(labels_array.sum())
            total += len(labels_array)

            weights = None if family_weights is None else family_weights.get(family)
            weight_tensors[family] = (
                torch.ones(len(labels_array), dtype=torch.float32) if weights is None
                else torch.from_numpy(np.asarray(weights, dtype=np.float32)))

            rows = np.arange(len(labels_array))
            groups = None if family_groups is None else family_groups.get(family)
            groups_array = None if groups is None else np.asarray(groups)
            family_train, family_val = self._split_rows(rows, labels_array, groups_array)

            mean, std = streaming_token_moments(flat_features, offsets_array, family_train)
            self.moments[family] = (mean.to(self.device), std.to(self.device))

            lengths = offsets_array[1:] - offsets_array[:-1]
            for batch in pack_length_sorted_batches(lengths[family_train].tolist(),
                                                    self.batch_token_budget):
                train_batches.append((family, [int(family_train[p]) for p in batch]))
            for batch in pack_length_sorted_batches(lengths[family_val].tolist(),
                                                    self.batch_token_budget):
                val_batches.append((family, [int(family_val[p]) for p in batch]))

        hidden_dims = {family: int(data[0].shape[1]) for family, data in family_data.items()}
        self.model = MultiFamilyTokenProbe(hidden_dims).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        positive_weight = torch.tensor(
            [(total - positives) / max(1, positives)], device=self.device)
        # reduction="none" so per-row weights can be applied before averaging;
        # with all-ones weights this is identical to the previous mean loss.
        deception_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight, reduction="none")
        unweighted_loss = nn.BCEWithLogitsLoss()
        generator = np.random.default_rng(self.seed)

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        for _ in range(self.max_epochs):
            self.model.train()
            for batch_position in generator.permutation(len(train_batches)):
                family, row_ids = train_batches[batch_position]
                flat_features, offsets, _ = family_data[family]
                batch_features, batch_mask = self._build_batch(
                    family, flat_features, np.asarray(offsets, dtype=np.int64), row_ids)
                optimizer.zero_grad()
                logits = self.model(family, batch_features, batch_mask)
                per_row = deception_loss(logits, label_tensors[family][row_ids].to(self.device))
                row_weights = weight_tensors[family][row_ids].to(self.device)
                loss = (per_row * row_weights).sum() / row_weights.sum().clamp(min=1e-8)
                loss.backward()
                optimizer.step()

            if not val_batches:
                continue
            self.model.eval()
            weighted_losses, n_val = [], 0
            with torch.no_grad():
                for family, row_ids in val_batches:
                    flat_features, offsets, _ = family_data[family]
                    batch_features, batch_mask = self._build_batch(
                        family, flat_features, np.asarray(offsets, dtype=np.int64), row_ids)
                    logits = self.model(family, batch_features, batch_mask)
                    batch_loss = unweighted_loss(
                        logits, label_tensors[family][row_ids].to(self.device))
                    weighted_losses.append(batch_loss.item() * len(row_ids))
                    n_val += len(row_ids)
            val_loss = sum(weighted_losses) / n_val
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {key: value.detach().clone()
                              for key, value in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, family: str, flat_features: torch.Tensor,
                      offsets: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        :param family: Which input projection to apply.
        :param flat_features: (total_tokens, hidden) tensor on self.device.
        :param offsets: (N+1,) int64 offsets into flat_features.
        :return: (N, 2) class probabilities, sklearn layout, rows in the
            original example order regardless of batch packing order.
        """
        offsets_array = np.asarray(offsets, dtype=np.int64)
        num_rows = len(offsets_array) - 1
        lengths = (offsets_array[1:] - offsets_array[:-1]).tolist()

        self.model.eval()
        positive_scores = np.zeros(num_rows, dtype=np.float64)
        with torch.no_grad():
            for row_ids in pack_length_sorted_batches(lengths, self.batch_token_budget):
                batch_features, batch_mask = self._build_batch(
                    family, flat_features, offsets_array, row_ids)
                probabilities = torch.sigmoid(self.model(family, batch_features, batch_mask))
                positive_scores[row_ids] = probabilities.detach().cpu().numpy()
        return np.stack([1.0 - positive_scores, positive_scores], axis=1)
