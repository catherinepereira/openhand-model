"""
Dataset for the ASL Fingerspelling sequences.

Loads pre-extracted .npz files produced by scripts/preprocess_fingerspelling.py.
Each file has the per-frame landmark tensor, an explicit per-landmark missing
mask, and the target sequence as int indices.

ctc_collate pads variable-length sequences within a batch and returns the
shapes CTCLoss expects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .landmarks import GROUP_OFFSETS, N_FEATURES, N_LANDMARKS, normalize_sequence


_LANDMARK_GROUPS: dict[str, tuple[int, int]] = {
    "face":       (GROUP_OFFSETS["face"]       // 3, GROUP_OFFSETS["pose"]       // 3),
    "pose":       (GROUP_OFFSETS["pose"]       // 3, GROUP_OFFSETS["left_hand"]  // 3),
    "left_hand":  (GROUP_OFFSETS["left_hand"]  // 3, GROUP_OFFSETS["right_hand"] // 3),
    "right_hand": (GROUP_OFFSETS["right_hand"] // 3, N_LANDMARKS),
}


class FingerspellingDataset(Dataset):
    def __init__(
        self,
        processed_dir: Path,
        sequences: pd.DataFrame,
        max_frames: int | None = 384,
        augment: bool = False,
        time_crop_min: float = 0.70,
        time_stretch_range: tuple[float, float] = (0.85, 1.15),
        time_mask_max_spans: int = 2,
        time_mask_max_len_frac: float = 0.10,
        frame_mask_prob: float = 0.05,
        group_dropout_prob: float = 0.08,
        affine_scale_range: tuple[float, float] = (0.9, 1.1),
        affine_translate: float = 0.05,
    ):
        self.seq_dir = Path(processed_dir) / "sequences"
        self.sequences = sequences.reset_index(drop=True)
        self.max_frames = max_frames
        self.augment = augment
        self.time_crop_min = time_crop_min
        self.time_stretch_range = time_stretch_range
        self.time_mask_max_spans = time_mask_max_spans
        self.time_mask_max_len_frac = time_mask_max_len_frac
        self.frame_mask_prob = frame_mask_prob
        self.group_dropout_prob = group_dropout_prob
        self.affine_scale_range = affine_scale_range
        self.affine_translate = affine_translate

    def __len__(self) -> int:
        return len(self.sequences)

    def _apply_augmentations(
        self, x: np.ndarray, missing: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        T = x.shape[0]
        rng = np.random

        if T > 4:
            keep_frac = rng.uniform(self.time_crop_min, 1.0)
            keep_len = max(4, int(T * keep_frac))
            start = rng.randint(0, T - keep_len + 1)
            x = x[start : start + keep_len]
            missing = missing[start : start + keep_len]
            T = x.shape[0]

        lo, hi = self.time_stretch_range
        if T > 4:
            stretch = rng.uniform(lo, hi)
            new_T = max(4, int(round(T * stretch)))
            if new_T != T:
                idx = np.linspace(0, T - 1, new_T).astype(int)
                x = x[idx]
                missing = missing[idx]
                T = new_T

        if self.group_dropout_prob > 0:
            for start_lm, end_lm in _LANDMARK_GROUPS.values():
                if rng.random() < self.group_dropout_prob:
                    f_start, f_end = start_lm * 3, end_lm * 3
                    x[:, f_start:f_end] = 0.0
                    missing[:, start_lm:end_lm] = True

        return x, missing

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        row = self.sequences.iloc[idx]
        seq_id = int(row["sequence_id"])
        path = self.seq_dir / f"{seq_id}.npz"
        try:
            data = np.load(path)
            x = data["x"].copy()
            missing = data["missing"].copy()
            target = data["target"].astype(np.int64)
        except FileNotFoundError:
            return torch.zeros(0, N_FEATURES), torch.zeros(0, dtype=torch.long), 0, 0

        if self.augment:
            x, missing = self._apply_augmentations(x, missing)

        if self.max_frames is not None and x.shape[0] > self.max_frames:
            keep = np.linspace(0, x.shape[0] - 1, self.max_frames).astype(int)
            x = x[keep]
            missing = missing[keep]

        x = normalize_sequence(x, missing)

        if self.augment and x.shape[0] > 0:
            lo, hi = self.affine_scale_range
            scale = np.random.uniform(lo, hi)
            tx = np.random.uniform(-self.affine_translate, self.affine_translate)
            ty = np.random.uniform(-self.affine_translate, self.affine_translate)
            pts = x.reshape(x.shape[0], -1, 3)
            pts[..., 0] = pts[..., 0] * scale + tx
            pts[..., 1] = pts[..., 1] * scale + ty
            pts[..., 2] = pts[..., 2] * scale
            # Re-zero absent landmarks; normalize_sequence's zeroing was undone by the affine.
            pts[missing] = 0.0
            x = pts.reshape(x.shape[0], -1)

        if self.augment and self.frame_mask_prob > 0 and x.shape[0] > 0:
            zap = np.random.rand(x.shape[0]) < self.frame_mask_prob
            x[zap] = 0.0

        if self.augment and self.time_mask_max_spans > 0 and x.shape[0] >= 8:
            n_spans = np.random.randint(0, self.time_mask_max_spans + 1)
            for _ in range(n_spans):
                span_len = max(1, int(np.random.uniform(0, self.time_mask_max_len_frac) * x.shape[0]))
                if span_len >= x.shape[0]:
                    continue
                start = np.random.randint(0, x.shape[0] - span_len + 1)
                x[start : start + span_len] = 0.0

        x_t = torch.from_numpy(x.astype(np.float32, copy=False))
        y_t = torch.from_numpy(target)
        return x_t, y_t, x_t.shape[0], y_t.shape[0]


def ctc_collate(batch):
    """Pad variable-length sequences within a batch.

    Returns:
      x_padded:        (B, T_max, N_FEATURES) float32
      y_concat:        (sum(target_lengths),) int64, flattened targets per CTC docs
      input_lengths:   (B,)
      target_lengths:  (B,)
      pad_mask:        (B, T_max) bool, True at padded positions
    """
    batch = [b for b in batch if b[2] > 0 and b[3] > 0]
    if not batch:
        return None

    xs, ys, in_lens, tgt_lens = zip(*batch)
    B = len(xs)
    T_max = max(in_lens)
    feat = xs[0].shape[-1]

    x_padded = torch.zeros(B, T_max, feat, dtype=torch.float32)
    pad_mask = torch.ones(B, T_max, dtype=torch.bool)
    for i, (x, t) in enumerate(zip(xs, in_lens)):
        x_padded[i, :t] = x
        pad_mask[i, :t] = False

    y_concat = torch.cat(list(ys), dim=0)
    input_lengths = torch.tensor(in_lens, dtype=torch.long)
    target_lengths = torch.tensor(tgt_lens, dtype=torch.long)
    return x_padded, y_concat, input_lengths, target_lengths, pad_mask


def load_char_vocab(raw_dir: Path) -> tuple[dict[str, int], dict[int, str], int]:
    """Load the character vocabulary. ``num_classes`` includes the CTC blank
    appended at index ``num_classes - 1``."""
    with open(Path(raw_dir) / "character_to_prediction_index.json") as f:
        char_to_idx: dict[str, int] = json.load(f)
    n_chars = len(char_to_idx)
    blank_idx = n_chars
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    idx_to_char[blank_idx] = "<blank>"
    return char_to_idx, idx_to_char, n_chars + 1
