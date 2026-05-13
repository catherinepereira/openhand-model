"""
Transformer encoder + CTC head for ASL fingerspelling sequence transcription.

Input:  (B, T, N_FEATURES) float32 — landmark sequences, padded to max T
        in batch
Output: (T, B, V) log-probabilities for CTC, where V = num_chars + 1 (blank)

Layout matches torch.nn.CTCLoss expectations (time-first).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .landmarks import N_FEATURES


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class CTCTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,            # includes the CTC blank as the last index
        n_features: int = N_FEATURES,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        conv_kernel: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Light 1D-conv stem over the feature axis: smooths jittery frames
        # before attention. (B, N_FEATURES, T) ↔ Conv1d expects (N, C, L).
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
        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,                # (B, T, N_FEATURES)
        src_key_padding_mask: torch.Tensor | None = None,  # (B, T) bool, True = PAD
    ) -> torch.Tensor:
        # Conv stem in (B, C, T)
        x = x.transpose(1, 2)              # (B, N_FEATURES, T)
        x = self.stem(x)                   # (B, d_model, T)
        x = x.transpose(1, 2)              # (B, T, d_model)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.head(x)              # (B, T, V)
        # CTCLoss wants (T, B, V) log-probs
        return F.log_softmax(logits, dim=-1).transpose(0, 1)


def build_ctc_model(num_classes: int, **kwargs) -> CTCTransformer:
    return CTCTransformer(num_classes=num_classes, **kwargs)
