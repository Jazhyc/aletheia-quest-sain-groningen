"""
Organism-invariant variant of the transformer token probe.

The shipped probe latches onto features that are specific to the training
organisms. On an unseen organism those features can carry no signal, or carry
it backwards -- `sonic_v1` scored AUROC 0.3526 on Notus/gemma, i.e. confidently
inverted. This probe attacks that directly with a domain-adversarial head
(Ganin et al., gradient reversal): a second classifier tries to predict which
organism a pooled representation came from, and the gradient reversal layer
makes the encoder maximize that classifier's loss. The encoder is therefore
pushed toward a representation from which the organism cannot be recovered,
while the deception head still has to work.

Architecture, optimizer, batching, standardization and early stopping are
identical to `TokenProbe("transformer", ...)`, so any difference in held-out
AUROC is attributable to the adversarial term alone.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from token_probes import (TransformerTokenProbe, pack_length_sorted_batches,
                          sinusoidal_position_encoding, streaming_token_moments)


class GradientReversal(torch.autograd.Function):
    """Identity forward, sign-flipped and scaled backward."""

    @staticmethod
    def forward(ctx, features: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


def pooled_representation(model: TransformerTokenProbe, padded_tokens: torch.Tensor,
                          padding_mask: torch.Tensor) -> torch.Tensor:
    """
    Run a :class:`TransformerTokenProbe` up to its masked mean pooling, without
    applying the classification head.

    :param model: The transformer token probe whose encoder to run.
    :param padded_tokens: (batch, max_len, hidden) standardized activations.
    :param padding_mask: (batch, max_len) bool, True marks a real token.
    :return: (batch, d_model) pooled representation.
    """
    sequence_length = padded_tokens.shape[1]
    position_encoding = sinusoidal_position_encoding(
        sequence_length, model.d_model, device=padded_tokens.device)
    tokens = model.projection(padded_tokens) + position_encoding.unsqueeze(0)
    encoded = model.encoder(tokens, src_key_padding_mask=~padding_mask)
    mask = padding_mask.unsqueeze(-1).to(encoded.dtype)
    return (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class InvariantTokenProbe:
    """
    Transformer token probe trained with a domain-adversarial organism head.

    Exposes the same ``fit`` / ``predict_proba`` contract as
    :class:`token_probes.TokenProbe`, except that ``fit`` also takes per-row
    organism ids. The adversary exists only during training; ``predict_proba``
    runs the deception head alone, so a fitted probe is a drop-in replacement.
    """

    def __init__(self, seed: int = 0, device: str | None = None, max_epochs: int = 60,
                 batch_token_budget: int = 8192, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, patience: int = 6,
                 validation_fraction: float = 0.15, adversary_strength: float = 1.0,
                 adversary_hidden: int = 128):
        """
        :param seed: Seed for weight init, the validation split, and batch order.
        :param device: Torch device; auto-selects cuda when available.
        :param max_epochs: Upper bound on training epochs.
        :param batch_token_budget: Max batch_size x max_len_in_batch per batch.
        :param learning_rate: Adam learning rate.
        :param weight_decay: Adam weight decay.
        :param patience: Epochs without validation improvement before stopping.
        :param validation_fraction: Fraction of training rows held out for
            early stopping.
        :param adversary_strength: Peak gradient reversal strength. The schedule
            ramps to this value over training, so early epochs fit the deception
            head before invariance pressure is applied.
        :param adversary_hidden: Width of the organism classifier's hidden layer.
        """
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs = max_epochs
        self.batch_token_budget = batch_token_budget
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.adversary_strength = adversary_strength
        self.adversary_hidden = adversary_hidden
        self.model: TransformerTokenProbe | None = None
        self.adversary: nn.Module | None = None
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None

    def _build_batch(self, flat_features: torch.Tensor, offsets: np.ndarray,
                     row_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param flat_features: (total_tokens, hidden) tensor on self.device.
        :param offsets: (N+1,) int64 offsets into flat_features.
        :param row_ids: Example indices to pack into this batch.
        :return: (standardized_padded_tokens, padding_mask) on self.device.
        """
        lengths = [int(offsets[row + 1] - offsets[row]) for row in row_ids]
        max_len = max(lengths)
        hidden_dim = flat_features.shape[1]
        padded = torch.zeros((len(row_ids), max_len, hidden_dim), dtype=torch.float32,
                             device=self.device)
        mask = torch.zeros((len(row_ids), max_len), dtype=torch.bool, device=self.device)
        for position, row in enumerate(row_ids):
            start, end = int(offsets[row]), int(offsets[row + 1])
            padded[position, :end - start] = flat_features[start:end].to(torch.float32)
            mask[position, :end - start] = True
        standardized = (padded - self.feature_mean) / self.feature_std
        return standardized * mask.unsqueeze(-1), mask

    def fit(self, flat_features: torch.Tensor, offsets: np.ndarray | torch.Tensor,
            labels: np.ndarray, domains: np.ndarray) -> "InvariantTokenProbe":
        """
        :param flat_features: (total_tokens, hidden) tensor already on
            self.device, all examples' response tokens concatenated in order.
        :param offsets: (N+1,) int64 offsets; example i's tokens are
            flat_features[offsets[i]:offsets[i + 1]].
        :param labels: (N,) binary deception labels.
        :param domains: (N,) organism identifiers; any hashable dtype. These are
            mapped to contiguous class ids for the adversary.
        :return: self, with the best-validation-loss weights loaded.
        """
        from sklearn.model_selection import train_test_split

        torch.manual_seed(self.seed)
        offsets_array = np.asarray(offsets, dtype=np.int64)
        labels_array = np.asarray(labels)
        _, domain_ids = np.unique(np.asarray(domains), return_inverse=True)
        n_domains = int(domain_ids.max()) + 1
        # With a single training organism the adversary's cross-entropy is
        # constant and its gradient is zero, so the fit silently degenerates to
        # the plain probe. Say so rather than reporting a no-op as a result.
        self.adversary_active = n_domains > 1
        if not self.adversary_active:
            self.adversary_strength = 0.0

        row_indices = np.arange(len(labels_array))
        train_rows, val_rows = train_test_split(
            row_indices, test_size=self.validation_fraction, random_state=self.seed,
            stratify=labels_array)

        mean, std = streaming_token_moments(flat_features, offsets_array, train_rows)
        self.feature_mean = mean.to(self.device)
        self.feature_std = std.to(self.device)

        lengths = offsets_array[1:] - offsets_array[:-1]
        train_batches = [[int(train_rows[position]) for position in batch]
                         for batch in pack_length_sorted_batches(
                             lengths[train_rows].tolist(), self.batch_token_budget)]
        val_batches = [[int(val_rows[position]) for position in batch]
                       for batch in pack_length_sorted_batches(
                           lengths[val_rows].tolist(), self.batch_token_budget)]

        label_tensor = torch.from_numpy(labels_array.astype(np.float32))
        domain_tensor = torch.from_numpy(domain_ids.astype(np.int64))

        self.model = TransformerTokenProbe(flat_features.shape[1]).to(self.device)
        self.adversary = nn.Sequential(
            nn.Linear(self.model.d_model, self.adversary_hidden),
            nn.ReLU(),
            nn.Linear(self.adversary_hidden, n_domains)).to(self.device)

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.adversary.parameters()),
            lr=self.learning_rate, weight_decay=self.weight_decay)
        deception_loss = nn.BCEWithLogitsLoss()
        organism_loss = nn.CrossEntropyLoss()
        generator = np.random.default_rng(self.seed)

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        for epoch in range(self.max_epochs):
            # Ganin's schedule: no reversal pressure at first, ramping to the
            # configured strength, so the deception head is not fighting an
            # adversary before it has learned anything.
            progress = epoch / max(1, self.max_epochs - 1)
            strength = self.adversary_strength * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)

            self.model.train()
            self.adversary.train()
            for batch_position in generator.permutation(len(train_batches)):
                row_ids = train_batches[batch_position]
                batch_features, batch_mask = self._build_batch(
                    flat_features, offsets_array, row_ids)
                pooled = pooled_representation(self.model, batch_features, batch_mask)
                logits = self.model.head(pooled).squeeze(-1)
                organism_logits = self.adversary(GradientReversal.apply(pooled, strength))
                optimizer.zero_grad()
                loss = (deception_loss(logits, label_tensor[row_ids].to(self.device))
                        + organism_loss(organism_logits, domain_tensor[row_ids].to(self.device)))
                loss.backward()
                optimizer.step()

            # Model selection uses the deception loss ONLY. Including the
            # adversarial term would let the encoder win by degrading the
            # detector, which is the opposite of what we are selecting for.
            self.model.eval()
            weighted_losses = []
            with torch.no_grad():
                for row_ids in val_batches:
                    batch_features, batch_mask = self._build_batch(
                        flat_features, offsets_array, row_ids)
                    logits = self.model(batch_features, batch_mask)
                    batch_loss = deception_loss(logits, label_tensor[row_ids].to(self.device))
                    weighted_losses.append(batch_loss.item() * len(row_ids))
            val_loss = sum(weighted_losses) / len(val_rows)
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

    def predict_proba(self, flat_features: torch.Tensor,
                      offsets: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        :param flat_features: (total_tokens, hidden) tensor on self.device.
        :param offsets: (N+1,) int64 offsets into flat_features.
        :return: (N, 2) class probabilities, sklearn layout, rows in the
            original example order regardless of batch packing order.
        """
        offsets_array = np.asarray(offsets, dtype=np.int64)
        num_rows = len(offsets_array) - 1
        lengths = (offsets_array[1:] - offsets_array[:-1]).tolist()
        batches = pack_length_sorted_batches(lengths, self.batch_token_budget)

        self.model.eval()
        positive_scores = np.zeros(num_rows, dtype=np.float64)
        with torch.no_grad():
            for row_ids in batches:
                batch_features, batch_mask = self._build_batch(
                    flat_features, offsets_array, row_ids)
                probabilities = torch.sigmoid(self.model(batch_features, batch_mask))
                positive_scores[row_ids] = probabilities.detach().cpu().numpy()
        return np.stack([1.0 - positive_scores, positive_scores], axis=1)
