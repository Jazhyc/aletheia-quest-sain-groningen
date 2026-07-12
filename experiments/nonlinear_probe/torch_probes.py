"""
Torch probe architectures reading the cached layer stack per example.

Each probe consumes a (layers, hidden) matrix — one pooled residual-stream
vector per cached decoder layer — so it can learn cross-layer structure
that the single-layer probes in nonlinear_sweep.py cannot. `TorchProbe`
wraps training behind the same fit/predict_proba interface sklearn
pipelines expose, so the sweep code treats every probe alike.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CNNLayerProbe(nn.Module):
    """
    1D CNN over the layer axis: two convolutions mixing neighbouring
    layers' features, global average pooling, and a linear head.
    """

    def __init__(self, hidden_dim: int, channels: int = 128, kernel_size: int = 3,
                 dropout: float = 0.1):
        """
        :param hidden_dim: Residual-stream width (input channels).
        :param channels: Convolution channels after the first layer.
        :param kernel_size: Convolution width along the layer axis.
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

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        """
        :param stack: (batch, layers, hidden) standardized activations.
        :return: (batch,) classification logits.
        """
        encoded = self.encoder(stack.transpose(1, 2))
        pooled = encoded.mean(dim=2)
        return self.head(pooled).squeeze(-1)


class TransformerLayerProbe(nn.Module):
    """
    Transformer encoder over the layer axis: each layer's pooled vector is
    projected to a token, tagged with a learned layer embedding, passed
    through self-attention blocks, mean-pooled, and classified.
    """

    def __init__(self, hidden_dim: int, n_layers: int, d_model: int = 128,
                 n_heads: int = 4, n_blocks: int = 2, dropout: float = 0.1):
        """
        :param hidden_dim: Residual-stream width per layer token.
        :param n_layers: Number of layer tokens in the stack.
        :param d_model: Transformer width after projection.
        :param n_heads: Attention heads per block.
        :param n_blocks: Encoder blocks.
        :param dropout: Dropout inside the encoder and before the head.
        """
        super().__init__()
        self.projection = nn.Linear(hidden_dim, d_model)
        self.layer_embedding = nn.Parameter(torch.zeros(1, n_layers, d_model))
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=n_blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        """
        :param stack: (batch, layers, hidden) standardized activations.
        :return: (batch,) classification logits.
        """
        tokens = self.projection(stack) + self.layer_embedding
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def build_probe_model(architecture: str, hidden_dim: int, n_layers: int) -> nn.Module:
    """
    :param architecture: 'cnn' or 'transformer'.
    :param hidden_dim: Residual-stream width.
    :param n_layers: Number of layers in the stack.
    :return: An untrained probe module.
    :raises ValueError: On an unknown architecture name.
    """
    if architecture == "cnn":
        return CNNLayerProbe(hidden_dim)
    if architecture == "transformer":
        return TransformerLayerProbe(hidden_dim, n_layers)
    raise ValueError(f"unknown architecture {architecture!r}")


def streaming_moments(features: np.ndarray, chunk_rows: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-(layer, dim) mean and standard deviation without materializing a
    float32 copy of the whole float16 feature array.

    :param features: (N, layers, hidden) float16 array.
    :param chunk_rows: Rows converted to float64 per accumulation step.
    :return: (mean, std) float32 arrays of shape (layers, hidden); std has
        a small floor to avoid division by zero on constant features.
    """
    total = np.zeros(features.shape[1:], dtype=np.float64)
    total_squared = np.zeros(features.shape[1:], dtype=np.float64)
    for start in range(0, len(features), chunk_rows):
        chunk = features[start:start + chunk_rows].astype(np.float64)
        total += chunk.sum(axis=0)
        total_squared += (chunk ** 2).sum(axis=0)
    mean = total / len(features)
    variance = np.maximum(total_squared / len(features) - mean ** 2, 0.0)
    std = np.maximum(np.sqrt(variance), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


class TorchProbe:
    """
    sklearn-style wrapper: standardizes the layer stack, trains with Adam
    and early stopping on a held-out validation split, and restores the
    best-validation-loss weights before scoring.
    """

    def __init__(self, architecture: str, seed: int = 0, device: str | None = None,
                 max_epochs: int = 60, batch_size: int = 128, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, patience: int = 6,
                 validation_fraction: float = 0.15):
        """
        :param architecture: 'cnn' or 'transformer' (see build_probe_model).
        :param seed: Seed for weight init and the validation split.
        :param device: Torch device; auto-selects cuda when available.
        :param max_epochs: Upper bound on training epochs.
        :param batch_size: Minibatch size.
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
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.model: nn.Module | None = None
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None

    def _standardize(self, batch: torch.Tensor) -> torch.Tensor:
        """
        :param batch: (batch, layers, hidden) float16/float32 tensor on device.
        :return: Standardized float32 tensor.
        """
        return (batch.to(torch.float32) - self.feature_mean) / self.feature_std

    def _epoch_batches(self, features: torch.Tensor, labels: torch.Tensor,
                       order: torch.Tensor):
        """
        :param features: (N, layers, hidden) float16 tensor (cpu).
        :param labels: (N,) float32 tensor (cpu).
        :param order: Row order for this epoch.
        :return: Generator of standardized (batch_features, batch_labels)
            pairs on the training device.
        """
        for start in range(0, len(order), self.batch_size):
            index = order[start:start + self.batch_size]
            batch = features[index].to(self.device)
            yield self._standardize(batch), labels[index].to(self.device)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "TorchProbe":
        """
        :param features: (N, layers, hidden) float16 training stacks.
        :param labels: (N,) binary training labels.
        :return: self, with the best-validation-loss weights loaded.
        """
        from sklearn.model_selection import train_test_split

        torch.manual_seed(self.seed)
        mean, std = streaming_moments(features)
        self.feature_mean = torch.from_numpy(mean).to(self.device)
        self.feature_std = torch.from_numpy(std).to(self.device)

        row_indices = np.arange(len(labels))
        train_rows, val_rows = train_test_split(
            row_indices, test_size=self.validation_fraction, random_state=self.seed,
            stratify=labels)
        feature_tensor = torch.from_numpy(features)
        label_tensor = torch.from_numpy(labels.astype(np.float32))

        self.model = build_probe_model(
            self.architecture, features.shape[2], features.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        loss_function = nn.BCEWithLogitsLoss()
        generator = np.random.default_rng(self.seed)

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        for _ in range(self.max_epochs):
            self.model.train()
            order = torch.from_numpy(generator.permutation(train_rows))
            for batch_features, batch_labels in self._epoch_batches(
                    feature_tensor, label_tensor, order):
                optimizer.zero_grad()
                loss = loss_function(self.model(batch_features), batch_labels)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_features, batch_labels in self._epoch_batches(
                        feature_tensor, label_tensor, torch.from_numpy(val_rows)):
                    val_losses.append(
                        loss_function(self.model(batch_features), batch_labels).item()
                        * len(batch_labels))
            val_loss = sum(val_losses) / len(val_rows)
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

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        :param features: (N, layers, hidden) float16 stacks to score.
        :return: (N, 2) class probabilities, sklearn layout.
        """
        feature_tensor = torch.from_numpy(features)
        self.model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                batch = feature_tensor[start:start + self.batch_size].to(self.device)
                logits = self.model(self._standardize(batch))
                scores.append(torch.sigmoid(logits).cpu().numpy())
        positive = np.concatenate(scores)
        return np.column_stack([1.0 - positive, positive])
