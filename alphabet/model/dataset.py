"""
PyTorch Dataset wrapping the preprocessed .npy files.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ASLDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.augment:
            x = self._augment(x)
        return x, self.y[idx]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # Landmarks are stored in raw camera frame, so the MLP is rotation-sensitive
        # at inference time. Heavier z (in-plane) jitter than x/y reflects how users
        # actually tilt their wrist.
        pts = x.view(21, 3)
        theta_z = (torch.rand(1).item() - 0.5) * 2 * math.radians(40)
        theta_x = (torch.rand(1).item() - 0.5) * 2 * math.radians(10)
        theta_y = (torch.rand(1).item() - 0.5) * 2 * math.radians(10)
        R = _rotation_matrix(theta_x, theta_y, theta_z)
        pts = pts @ R.T
        x = pts.reshape(63)
        x = x + torch.randn_like(x) * 0.01
        scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.1
        return x * scale


def _rotation_matrix(rx: float, ry: float, rz: float) -> torch.Tensor:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
    return Rz @ Ry @ Rx


def load_splits(
    processed_dir: Path,
    val_frac: float = 0.15,
    test_frac: float = 0.05,
    seed: int = 42,
) -> tuple["ASLDataset", "ASLDataset", "ASLDataset"]:
    from sklearn.model_selection import train_test_split

    X = np.load(processed_dir / "X.npy")
    y = np.load(processed_dir / "y.npy")

    counts = np.bincount(y)
    keep_mask = counts[y] >= 2
    if not keep_mask.all():
        dropped = np.where(counts < 2)[0]
        print(f"Dropping {(~keep_mask).sum()} samples from singleton classes: {dropped.tolist()}")
        X, y = X[keep_mask], y[keep_mask]

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y
    )
    val_frac_adjusted = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac_adjusted, random_state=seed, stratify=y_tv
    )

    return (
        ASLDataset(X_train, y_train, augment=True),
        ASLDataset(X_val, y_val, augment=False),
        ASLDataset(X_test, y_test, augment=False),
    )
