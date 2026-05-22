"""
Squeezeformer-style encoder + CTC head for ASL fingerspelling.

Input:  (B, T, N_FEATURES) float32 landmark sequences, padded to max T in batch
Output: (T, B, V) log-probabilities for CTC, V = num_chars + 1 (blank)

Blocks follow the Squeezeformer ordering (FFN -> MHSA -> FFN -> ConvModule)
with a depthwise conv kernel of 51. The temporal squeeze/unsqueeze from the
original Squeezeformer paper is dropped: fingerspelling sequences are ~90
frames, short enough that the U-net structure doesn't pay off here.
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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FeedForwardModule(nn.Module):
    """Macaron-style FFN with 0.5x residual scaling."""
    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + 0.5 * h


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x)
        # need_weights=False keeps the ONNX graph simpler and runs faster.
        h, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = self.dropout(h)
        return x + h


class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)            # (B, C, T)
        h = self.pointwise1(h)                       # (B, 2C, T)
        h = F.glu(h, dim=1)                          # (B, C, T)
        h = self.depthwise(h)
        h = self.bn(h)
        h = self.act(h)
        h = self.pointwise2(h)
        h = self.dropout(h)
        return x + h.transpose(1, 2)


class SqueezeformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        conv_kernel: int,
        dropout: float,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ff_dim, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, n_heads, dropout)
        self.ffn2 = FeedForwardModule(d_model, ff_dim, dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.mhsa(x, key_padding_mask=key_padding_mask)
        x = self.ffn2(x)
        x = self.conv(x)
        return self.final_norm(x)


class CTCTransformer(nn.Module):
    # Class name kept for backwards compatibility with train_ctc.py and
    # export_ctc_onnx.py, which import it directly. The encoder is no longer
    # a plain Transformer.
    def __init__(
        self,
        num_classes: int,
        n_features: int = N_FEATURES,
        d_model: int = 144,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 576,
        dropout: float = 0.1,
        conv_kernel: int = 51,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Conv stem gives the encoder local temporal context before the
        # global attention layers see the features.
        stem_kernel = 5
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=stem_kernel, padding=stem_kernel // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=stem_kernel, padding=stem_kernel // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.pos = SinusoidalPositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            SqueezeformerBlock(
                d_model=d_model,
                n_heads=nhead,
                ff_dim=dim_feedforward,
                conv_kernel=conv_kernel,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.transpose(1, 2)                # (B, F, T)
        x = self.stem(x)                      # (B, D, T)
        x = x.transpose(1, 2)                 # (B, T, D)
        x = self.pos(x)
        for block in self.blocks:
            x = block(x, key_padding_mask=src_key_padding_mask)
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)


def build_ctc_model(num_classes: int, **kwargs) -> CTCTransformer:
    return CTCTransformer(num_classes=num_classes, **kwargs)
