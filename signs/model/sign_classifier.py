"""
Isolated-sign classifier: 1D-Conv stem + Transformer encoder + masked
global-average-pooled classification head.

Pattern follows the Kaggle Google ISLR winners (Conv1D combined with
Transformer over the temporal axis, attention over frames not landmarks).
Roughly 3-5M params depending on d_model / layer count.

Input:  (B, T, N_FEATURES) float32 feature sequences + (B, T) pad mask
Output: (B, num_classes) logits (no softmax; let the loss handle it)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .signs_landmarks import N_FEATURES


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SignClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 250,
        n_features: int = N_FEATURES,
        d_model: int = 192,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        conv_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.pos = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,                # (B, T, N_FEATURES)
        pad_mask: torch.Tensor | None = None,  # (B, T) bool, True = pad
    ) -> torch.Tensor:
        x = x.transpose(1, 2)              # (B, N_FEATURES, T) for Conv1d
        x = self.stem(x)
        x = x.transpose(1, 2)              # (B, T, d_model)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # Masked mean-pool over time.
        if pad_mask is None:
            pooled = x.mean(dim=1)
        else:
            valid = (~pad_mask).unsqueeze(-1).float()
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        return self.head(pooled)


def build_sign_classifier(num_classes: int = 250, **kwargs) -> SignClassifier:
    return SignClassifier(num_classes=num_classes, **kwargs)


def encode_features(
    model: SignClassifier,
    x: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the pooled feature vector pre-head (B, d_model). Used by
    the medoid-clip builder to find each class's prototype clip in
    embedding space."""
    x = x.transpose(1, 2)
    x = model.stem(x)
    x = x.transpose(1, 2)
    x = model.pos(x)
    x = model.encoder(x, src_key_padding_mask=pad_mask)
    if pad_mask is None:
        return x.mean(dim=1)
    valid = (~pad_mask).unsqueeze(-1).float()
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
