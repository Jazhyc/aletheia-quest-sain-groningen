"""
Torch probe architectures reading per-token response activations.

Unlike torch_probes.py's fixed-size layer stack, the token caches hold a
variable number of response tokens per example, all pooled activations at
a single layer. Each probe consumes a (tokens, hidden) sequence directly —
an attention pooler, a small CNN, and a Transformer encoder — instead of the
mean/last pooling used upstream of the linear and MLP probes, so it can
learn which tokens matter. `TokenProbe` wraps training behind the same
fit/predict_proba interface the other probes expose, keeping the sweep code
uniform, but its data representation differs: examples are variable length,
so features are passed as a flat (total_tokens, hidden) tensor plus
per-example offsets rather than a dense (N, ...) array.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def sinusoidal_position_encoding(sequence_length: int, d_model: int,
                                 device: torch.device | str | None = None) -> torch.Tensor:
    """
    :param sequence_length: Number of token positions to encode.
    :param d_model: Encoding width (matches the projected token width).
    :param device: Torch device for the returned tensor.
    :return: (sequence_length, d_model) sinusoidal position encoding.
    """
    position = torch.arange(sequence_length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
                         * (-math.log(10000.0) / d_model))
    encoding = torch.zeros(sequence_length, d_model, device=device)
    encoding[:, 0::2] = torch.sin(position * div_term)
    cosine_columns = encoding[:, 1::2].shape[1]
    encoding[:, 1::2] = torch.cos(position * div_term)[:, :cosine_columns]
    return encoding


class AttentionTokenProbe(nn.Module):
    """
    Attention pooling over tokens: each token is projected and scored, a
    masked softmax over the token axis turns scores into pooling weights,
    and the weighted sum feeds a linear head.
    """

    def __init__(self, hidden_dim: int, d_model: int = 128, dropout: float = 0.1):
        """
        :param hidden_dim: Residual-stream width of each token.
        :param d_model: Width after the token projection.
        :param dropout: Dropout before the linear head.
        """
        super().__init__()
        self.projection = nn.Linear(hidden_dim, d_model)
        self.score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, padded_tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        :param padded_tokens: (batch, max_len, hidden) standardized activations.
        :param padding_mask: (batch, max_len) bool, True marks a real token.
        :return: (batch,) classification logits.
        """
        projected = self.projection(padded_tokens)
        scores = self.score(projected).squeeze(-1)
        scores = scores.masked_fill(~padding_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = (projected * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)


class CNNTokenProbe(nn.Module):
    """
    1D CNN over the token axis: two convolutions mixing neighbouring
    tokens' features, masked mean pooling, and a linear head.
    """

    def __init__(self, hidden_dim: int, channels: int = 128, kernel_size: int = 5,
                 dropout: float = 0.1):
        """
        :param hidden_dim: Residual-stream width (input channels).
        :param channels: Convolution channels after the first layer.
        :param kernel_size: Convolution width along the token axis.
        :param dropout: Dropout before the linear head.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding="same"),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 1))

    def forward(self, padded_tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        :param padded_tokens: (batch, max_len, hidden) standardized activations;
            padding positions are zero before the first conv.
        :param padding_mask: (batch, max_len) bool, True marks a real token.
        :return: (batch,) classification logits.

        The mask is applied after each convolution so that ``padding="same"``
        boundary effects at non-real positions do not propagate to the next
        conv layer.  The result is invariant to how much padding (zeros) is
        appended beyond the real tokens.
        """
        mask3d = padding_mask.unsqueeze(1).to(padded_tokens.dtype)        # (B, 1, L)
        x = padded_tokens.transpose(1, 2) * mask3d                        # (B, H, L), padding zeroed

        for i in range(0, len(self.encoder), 2):                          # conv, relu, conv, relu
            x = self.encoder[i](x)                                        # conv (padding="same" → shape unchanged)
            x = self.encoder[i + 1](x)                                    # relu
            x = x * mask3d                                                # kill boundary leakage

        pooled = x.sum(dim=2) / mask3d.sum(dim=2).clamp(min=1.0)          # (B, channels)
        return self.head(pooled).squeeze(-1)


class TransformerTokenProbe(nn.Module):
    """
    Transformer encoder over the token axis: tokens are projected, tagged
    with sinusoidal position encodings, passed through self-attention
    blocks that ignore padding via `src_key_padding_mask`, masked mean
    pooled, and classified.
    """

    def __init__(self, hidden_dim: int, d_model: int = 128, n_heads: int = 4,
                 dim_feedforward: int = 256, n_blocks: int = 2, dropout: float = 0.1):
        """
        :param hidden_dim: Residual-stream width per token.
        :param d_model: Transformer width after projection.
        :param n_heads: Attention heads per block.
        :param dim_feedforward: Feed-forward width inside each block.
        :param n_blocks: Encoder blocks.
        :param dropout: Dropout inside the encoder and before the head.
        """
        super().__init__()
        self.d_model = d_model
        self.projection = nn.Linear(hidden_dim, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=n_blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, padded_tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        :param padded_tokens: (batch, max_len, hidden) standardized activations.
        :param padding_mask: (batch, max_len) bool, True marks a real token.
        :return: (batch,) classification logits.
        """
        sequence_length = padded_tokens.shape[1]
        position_encoding = sinusoidal_position_encoding(
            sequence_length, self.d_model, device=padded_tokens.device)
        tokens = self.projection(padded_tokens) + position_encoding.unsqueeze(0)
        encoded = self.encoder(tokens, src_key_padding_mask=~padding_mask)
        mask = padding_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


def build_token_probe_model(architecture: str, hidden_dim: int) -> nn.Module:
    """
    :param architecture: 'attention', 'cnn', or 'transformer'.
    :param hidden_dim: Residual-stream width.
    :return: An untrained probe module.
    :raises ValueError: On an unknown architecture name.
    """
    if architecture == "attention":
        return AttentionTokenProbe(hidden_dim)
    if architecture == "cnn":
        return CNNTokenProbe(hidden_dim)
    if architecture == "transformer":
        return TransformerTokenProbe(hidden_dim)
    raise ValueError(f"unknown architecture {architecture!r}")


def pack_length_sorted_batches(lengths: list[int], token_budget: int) -> list[list[int]]:
    """
    Group example positions into length-sorted batches under a token budget.

    Peak memory for a padded batch scales with batch_size x max_len_in_batch,
    so the constraint is on that product, not the row count. A single example
    longer than the budget still becomes its own one-row batch.

    :param lengths: Token count of each example, indexed by position.
    :param token_budget: Maximum batch_size x max_len_in_batch per batch.
    :return: List of batches, each a list of positions into `lengths`; every
        position in range(len(lengths)) appears in exactly one batch.
    """
    order = sorted(range(len(lengths)), key=lambda position: lengths[position])
    batches: list[list[int]] = []
    current: list[int] = []
    for position in order:
        width = lengths[position]
        if current and (len(current) + 1) * width > token_budget:
            batches.append(current)
            current = []
        current.append(position)
    if current:
        batches.append(current)
    return batches


def streaming_token_moments(
        flat_features: torch.Tensor, offsets: np.ndarray, row_indices: np.ndarray,
        chunk_rows: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-dim mean and standard deviation over a subset of examples' tokens,
    without materializing a float32 copy of every token at once.

    :param flat_features: (total_tokens, hidden) tensor on the working device.
    :param offsets: (N+1,) int64 offsets; example i's tokens are
        flat_features[offsets[i]:offsets[i + 1]].
    :param row_indices: Example indices whose tokens contribute to the stats
        (e.g. the training split only).
    :param chunk_rows: Examples accumulated per chunk.
    :return: (mean, std) float32 tensors of shape (hidden,) on
        flat_features.device; std has a small floor to avoid division by zero.
    """
    hidden_dim = flat_features.shape[1]
    total = torch.zeros(hidden_dim, dtype=torch.float64, device=flat_features.device)
    total_squared = torch.zeros(hidden_dim, dtype=torch.float64, device=flat_features.device)
    total_tokens = 0
    for start in range(0, len(row_indices), chunk_rows):
        chunk_rows_indices = row_indices[start:start + chunk_rows]
        chunk_tokens = torch.cat(
            [flat_features[offsets[row]:offsets[row + 1]] for row in chunk_rows_indices], dim=0,
        ).to(torch.float64)
        total += chunk_tokens.sum(dim=0)
        total_squared += (chunk_tokens ** 2).sum(dim=0)
        total_tokens += chunk_tokens.shape[0]
    mean = total / total_tokens
    variance = torch.clamp(total_squared / total_tokens - mean ** 2, min=0.0)
    std = torch.clamp(variance.sqrt(), min=1e-6)
    return mean.to(torch.float32), std.to(torch.float32)


class TokenProbe:
    """
    sklearn-style wrapper around the token-sequence probes: standardizes
    tokens using training-set moments, packs variable-length examples into
    token-budgeted batches, trains with Adam and early stopping on a held-out
    validation split, and restores the best-validation-loss weights before
    scoring.
    """

    def __init__(self, architecture: str, seed: int = 0, device: str | None = None,
                 max_epochs: int = 60, batch_token_budget: int = 8192,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4, patience: int = 6,
                 validation_fraction: float = 0.15):
        """
        :param architecture: 'attention', 'cnn', or 'transformer' (see
            build_token_probe_model).
        :param seed: Seed for weight init, the validation split, and batch order.
        :param device: Torch device; auto-selects cuda when available.
        :param max_epochs: Upper bound on training epochs.
        :param batch_token_budget: Max batch_size x max_len_in_batch per batch.
        :param learning_rate: Adam learning rate.
        :param weight_decay: Adam weight decay.
        :param patience: Epochs without validation improvement before stopping.
        :param validation_fraction: Fraction of training rows held out for
            early stopping.
        """
        self.architecture = architecture
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs = max_epochs
        self.batch_token_budget = batch_token_budget
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.model: nn.Module | None = None
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None

    def _build_batch(self, flat_features: torch.Tensor, offsets: np.ndarray,
                     row_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param flat_features: (total_tokens, hidden) tensor on self.device.
        :param offsets: (N+1,) int64 offsets into flat_features.
        :param row_ids: Example indices to pack into this batch.
        :return: (standardized_padded_tokens, padding_mask) on self.device;
            padding positions are zeroed after standardization.
        """
        lengths = [int(offsets[row + 1] - offsets[row]) for row in row_ids]
        max_len = max(lengths)
        hidden_dim = flat_features.shape[1]
        padded = torch.zeros((len(row_ids), max_len, hidden_dim), dtype=torch.float32,
                             device=self.device)
        mask = torch.zeros((len(row_ids), max_len), dtype=torch.bool, device=self.device)
        for position, row in enumerate(row_ids):
            start, end = int(offsets[row]), int(offsets[row + 1])
            length = end - start
            padded[position, :length] = flat_features[start:end].to(torch.float32)
            mask[position, :length] = True
        standardized = (padded - self.feature_mean) / self.feature_std
        return standardized * mask.unsqueeze(-1), mask

    def fit(self, flat_features: torch.Tensor, offsets: np.ndarray | torch.Tensor,
            labels: np.ndarray) -> "TokenProbe":
        """
        :param flat_features: (total_tokens, hidden) float16 tensor already
            on self.device, all examples' response tokens concatenated in
            example order.
        :param offsets: (N+1,) int64 offsets; example i's tokens are
            flat_features[offsets[i]:offsets[i + 1]].
        :param labels: (N,) binary training labels.
        :return: self, with the best-validation-loss weights loaded.
        """
        from sklearn.model_selection import train_test_split

        torch.manual_seed(self.seed)
        offsets_array = np.asarray(offsets, dtype=np.int64)
        labels_array = np.asarray(labels)
        row_indices = np.arange(len(labels_array))
        train_rows, val_rows = train_test_split(
            row_indices, test_size=self.validation_fraction, random_state=self.seed,
            stratify=labels_array)

        mean, std = streaming_token_moments(flat_features, offsets_array, train_rows)
        self.feature_mean = mean.to(self.device)
        self.feature_std = std.to(self.device)

        lengths = offsets_array[1:] - offsets_array[:-1]
        train_batches = [
            [int(train_rows[position]) for position in batch]
            for batch in pack_length_sorted_batches(
                lengths[train_rows].tolist(), self.batch_token_budget)
        ]
        val_batches = [
            [int(val_rows[position]) for position in batch]
            for batch in pack_length_sorted_batches(
                lengths[val_rows].tolist(), self.batch_token_budget)
        ]
        label_tensor = torch.from_numpy(labels_array.astype(np.float32))

        self.model = build_token_probe_model(
            self.architecture, flat_features.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        loss_function = nn.BCEWithLogitsLoss()
        generator = np.random.default_rng(self.seed)

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        for _ in range(self.max_epochs):
            self.model.train()
            for batch_position in generator.permutation(len(train_batches)):
                row_ids = train_batches[batch_position]
                batch_features, batch_mask = self._build_batch(flat_features, offsets_array, row_ids)
                batch_labels = label_tensor[row_ids].to(self.device)
                optimizer.zero_grad()
                loss = loss_function(self.model(batch_features, batch_mask), batch_labels)
                loss.backward()
                optimizer.step()

            self.model.eval()
            weighted_losses = []
            with torch.no_grad():
                for row_ids in val_batches:
                    batch_features, batch_mask = self._build_batch(flat_features, offsets_array, row_ids)
                    batch_labels = label_tensor[row_ids].to(self.device)
                    loss = loss_function(self.model(batch_features, batch_mask), batch_labels)
                    weighted_losses.append(loss.item() * len(row_ids))
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

    def decision_function(self, flat_features: torch.Tensor,
                          offsets: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Raw pre-sigmoid probe scores, for ranking rather than calibration.

        ``predict_proba`` squashes through a float32 sigmoid, which saturates to
        exactly 1.0 above a logit of roughly 17 and ties every confident row
        together.  AUROC is computed from the ranking alone, so callers that
        only rank should read the log-odds here instead.

        :param flat_features: (total_tokens, hidden) tensor on self.device.
        :param offsets: (N+1,) int64 offsets into flat_features.
        :return: (N,) log-odds, rows in the original example order.
        """
        offsets_array = np.asarray(offsets, dtype=np.int64)
        num_rows = len(offsets_array) - 1
        lengths = (offsets_array[1:] - offsets_array[:-1]).tolist()
        batches = pack_length_sorted_batches(lengths, self.batch_token_budget)

        self.model.eval()
        logits = np.zeros(num_rows, dtype=np.float64)
        with torch.no_grad():
            for row_ids in batches:
                batch_features, batch_mask = self._build_batch(flat_features, offsets_array, row_ids)
                values = self.model(batch_features, batch_mask).float().cpu().numpy()
                for position, row in enumerate(row_ids):
                    logits[row] = values[position]
        return logits

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
                batch_features, batch_mask = self._build_batch(flat_features, offsets_array, row_ids)
                probabilities = torch.sigmoid(
                    self.model(batch_features, batch_mask)).cpu().numpy()
                for position, row in enumerate(row_ids):
                    positive_scores[row] = probabilities[position]
        return np.column_stack([1.0 - positive_scores, positive_scores])
