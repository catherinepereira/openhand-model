"""
PyTorch Dataset wrapping the preprocessed .npy files.
"""

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
        # Small Gaussian noise on landmark positions
        x = x + torch.randn_like(x) * 0.01
        # Random scale jitter ±5%
        scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.1
        x = x * scale
        return x


def load_splits(
    processed_dir: Path,
    val_frac: float = 0.15,
    test_frac: float = 0.05,
    seed: int = 42,
) -> tuple["ASLDataset", "ASLDataset", "ASLDataset"]:
    from sklearn.model_selection import train_test_split

    X = np.load(processed_dir / "X.npy")
    y = np.load(processed_dir / "y.npy")

    # Drop classes with fewer than 2 samples (can't stratify-split singletons)
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
