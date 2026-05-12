"""
ASL sign MLP classifier.

Input:  (batch, 63)  — 21 hand landmarks × (x, y, z), wrist-centered + scaled
Output: (batch, 26)  — logits over A-Z

Architecture: 3 hidden layers with BatchNorm + Dropout, chosen to be fast enough
for real-time CPU inference once exported to ONNX (~0.1ms per frame).
"""

import torch
import torch.nn as nn


class ASLClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 63,
        num_classes: int = 26,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(num_classes: int = 26) -> ASLClassifier:
    return ASLClassifier(num_classes=num_classes)
