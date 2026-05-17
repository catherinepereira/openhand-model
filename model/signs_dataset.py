"""
PyTorch Dataset for the isolated signs classifier.

Loads preprocessed per-clip .npz files (produced by
scripts/preprocess_signs.py), runs per-clip normalization, builds the
engineered feature stack, applies winners-style augmentation in training,
and pads to a fixed time length in the collate fn.

Returned batch shape:
  x:        (B, T_max, N_FEATURES) float32
  labels:   (B,) int64
  pad_mask: (B, T_max) bool, True at padded positions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .signs_landmarks import (
    LEFT_HAND_SLICE,
    N_FEATURES,
    N_LANDMARKS,
    RIGHT_HAND_SLICE,
    build_features,
    normalize_clip,
)


class SignsDataset(Dataset):
    """One sample = one isolated-sign clip + its class label."""

    def __init__(
        self,
        processed_dir: Path,
        clips: pd.DataFrame,
        max_frames: int = 80,
        augment: bool = False,
        # Augmentation knobs (winners' patterns, conservative defaults)
        time_crop_min: float = 0.7,
        time_stretch_range: tuple[float, float] = (0.85, 1.15),
        affine_rot_deg: float = 30.0,
        affine_scale_range: tuple[float, float] = (0.9, 1.1),
        affine_translate: float = 0.05,
        hand_swap_prob: float = 0.5,
        frame_mask_prob: float = 0.05,
        group_dropout_prob: float = 0.04,
    ) -> None:
        self.clips_dir = Path(processed_dir) / "clips"
        self.clips = clips.reset_index(drop=True)
        self.max_frames = max_frames
        self.augment = augment

        self.time_crop_min = time_crop_min
        self.time_stretch_range = time_stretch_range
        self.affine_rot_deg = affine_rot_deg
        self.affine_scale_range = affine_scale_range
        self.affine_translate = affine_translate
        self.hand_swap_prob = hand_swap_prob
        self.frame_mask_prob = frame_mask_prob
        self.group_dropout_prob = group_dropout_prob

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.clips.iloc[idx]
        seq_id = int(row["sequence_id"])
        label = int(row["label"])

        path = self.clips_dir / f"{seq_id}.npz"
        try:
            data = np.load(path)
            x = data["x"]               # (T, 127, 3)
            missing = data["missing"]   # (T, 127)
        except FileNotFoundError:
            return torch.zeros(0, N_FEATURES), label

        if x.shape[0] == 0:
            return torch.zeros(0, N_FEATURES), label

        if self.augment:
            x, missing = self._apply_augmentations(x, missing)

        # Cap length: stride-sample if too long.
        if x.shape[0] > self.max_frames:
            idxs = np.linspace(0, x.shape[0] - 1, self.max_frames).astype(int)
            x = x[idxs]
            missing = missing[idxs]

        x = normalize_clip(x, missing)
        features, _ = build_features(x, missing)
        return torch.from_numpy(features), label

    # ------------- augmentation helpers -------------

    def _apply_augmentations(
        self, x: np.ndarray, missing: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        T = x.shape[0]
        rng = np.random

        # Time crop: keep a contiguous span 70-100% of the clip.
        if T > 4:
            keep_frac = rng.uniform(self.time_crop_min, 1.0)
            keep_len = max(4, int(T * keep_frac))
            start = rng.randint(0, T - keep_len + 1)
            x = x[start : start + keep_len]
            missing = missing[start : start + keep_len]
            T = x.shape[0]

        # Time stretch: resample to ~0.85-1.15x length.
        lo, hi = self.time_stretch_range
        if T > 4:
            stretch = rng.uniform(lo, hi)
            new_T = max(4, int(round(T * stretch)))
            if new_T != T:
                idxs = np.linspace(0, T - 1, new_T).astype(int)
                x = x[idxs]
                missing = missing[idxs]
                T = new_T

        # Horizontal flip with hand-index swap (winners' standard).
        if rng.random() < self.hand_swap_prob:
            x = x.copy()
            x[..., 0] = -x[..., 0]  # flip x
            # Swap left and right hand slots so the dominant hand stays
            # correct after mirroring.
            left = x[:, LEFT_HAND_SLICE, :].copy()
            right = x[:, RIGHT_HAND_SLICE, :].copy()
            x[:, LEFT_HAND_SLICE, :] = right
            x[:, RIGHT_HAND_SLICE, :] = left
            left_m = missing[:, LEFT_HAND_SLICE].copy()
            right_m = missing[:, RIGHT_HAND_SLICE].copy()
            missing = missing.copy()
            missing[:, LEFT_HAND_SLICE] = right_m
            missing[:, RIGHT_HAND_SLICE] = left_m

        # Affine in 2D (x, y): random rotation + scale + translation.
        if self.affine_rot_deg > 0:
            theta = np.deg2rad(rng.uniform(-self.affine_rot_deg, self.affine_rot_deg))
            sc = rng.uniform(*self.affine_scale_range)
            tx = rng.uniform(-self.affine_translate, self.affine_translate)
            ty = rng.uniform(-self.affine_translate, self.affine_translate)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R = np.array(
                [[cos_t * sc, -sin_t * sc], [sin_t * sc, cos_t * sc]],
                dtype=np.float32,
            )
            xy = x[..., :2]  # (T, 127, 2)
            rotated = xy @ R.T
            rotated[..., 0] += tx
            rotated[..., 1] += ty
            x = x.copy()
            x[..., :2] = rotated
            # z scales with the 2D scale so depth stays consistent
            x[..., 2] = x[..., 2] * sc

        # Group dropout: occasionally zero out face or pose entirely.
        if self.group_dropout_prob > 0:
            for slot in (slice(0, 76), slice(76, 85)):  # face, pose
                if rng.random() < self.group_dropout_prob:
                    x = x.copy()
                    missing = missing.copy()
                    x[:, slot, :] = 0.0
                    missing[:, slot] = True

        # Per-frame mask (Bernoulli).
        if self.frame_mask_prob > 0 and x.shape[0] > 0:
            zap = rng.random(x.shape[0]) < self.frame_mask_prob
            x = x.copy()
            missing = missing.copy()
            x[zap] = 0.0
            missing[zap] = True

        return x, missing


def signs_collate(batch):
    """Pad variable-length feature sequences to T_max within the batch."""
    batch = [(x, lbl) for x, lbl in batch if x.shape[0] > 0]
    if not batch:
        return None
    xs, labels = zip(*batch)
    B = len(xs)
    T_max = max(x.shape[0] for x in xs)
    feat = xs[0].shape[-1]
    x_padded = torch.zeros(B, T_max, feat, dtype=torch.float32)
    pad_mask = torch.ones(B, T_max, dtype=torch.bool)
    for i, x in enumerate(xs):
        t = x.shape[0]
        x_padded[i, :t] = x
        pad_mask[i, :t] = False
    labels_t = torch.tensor(labels, dtype=torch.long)
    return x_padded, labels_t, pad_mask


def load_sign_map(processed_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Return the canonical sign-name <-> index maps."""
    import json
    with open(Path(processed_dir) / "sign_to_idx.json") as f:
        sign_to_idx: dict[str, int] = json.load(f)
    idx_to_sign = {v: k for k, v in sign_to_idx.items()}
    return sign_to_idx, idx_to_sign
