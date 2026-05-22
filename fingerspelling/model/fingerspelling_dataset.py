"""
Dataset for the ASL Fingerspelling sequences.

Loads pre-extracted .npz files produced by scripts/preprocess_fingerspelling.py.
Each file has the per-frame landmark tensor, an explicit per-landmark missing
mask, and the target sequence as int indices.

Augmentations are based on the 1st-place Kaggle solution
(github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution):
time resample, affine (scale/shear/translate/rotate), time + spatial masking,
horizontal flip with left/right swap, finger dropout, and group dropout.

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

_LEFT_HAND_LM_START  = GROUP_OFFSETS["left_hand"]  // 3
_RIGHT_HAND_LM_START = GROUP_OFFSETS["right_hand"] // 3
_HAND_LEN = 21

# MediaPipe hand: 21 landmarks per hand. Indices 1..4 are thumb, 5..8 index,
# 9..12 middle, 13..16 ring, 17..20 pinky. 0 is wrist.
_FINGERS: list[list[int]] = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]


class FingerspellingDataset(Dataset):
    def __init__(
        self,
        processed_dir: Path,
        sequences: pd.DataFrame,
        max_frames: int | None = 384,
        augment: bool = False,
        # 1st-place magnitudes
        time_resample_range: tuple[float, float] = (0.5, 1.5),
        time_resample_prob: float = 0.8,
        affine_prob: float = 0.75,
        affine_scale_range: tuple[float, float] = (0.8, 1.2),
        affine_shear_max: float = 0.15,
        affine_translate_max: float = 0.10,
        affine_rotation_deg_max: float = 30.0,
        time_mask_prob: float = 0.5,
        time_mask_frac_range: tuple[float, float] = (0.20, 0.40),
        spatial_mask_prob: float = 0.5,
        spatial_mask_frac_range: tuple[float, float] = (0.05, 0.10),
        hflip_prob: float = 0.5,
        finger_dropout_prob: float = 0.10,
        # Held over from the previous pipeline: occasionally drop entire
        # landmark groups (face/pose/hand) so the model sees real-world
        # cases where MediaPipe misses a whole group.
        group_dropout_prob: float = 0.05,
    ):
        self.seq_dir = Path(processed_dir) / "sequences"
        self.sequences = sequences.reset_index(drop=True)
        self.max_frames = max_frames
        self.augment = augment

        self.time_resample_range = time_resample_range
        self.time_resample_prob = time_resample_prob
        self.affine_prob = affine_prob
        self.affine_scale_range = affine_scale_range
        self.affine_shear_max = affine_shear_max
        self.affine_translate_max = affine_translate_max
        self.affine_rotation_deg_max = affine_rotation_deg_max
        self.time_mask_prob = time_mask_prob
        self.time_mask_frac_range = time_mask_frac_range
        self.spatial_mask_prob = spatial_mask_prob
        self.spatial_mask_frac_range = spatial_mask_frac_range
        self.hflip_prob = hflip_prob
        self.finger_dropout_prob = finger_dropout_prob
        self.group_dropout_prob = group_dropout_prob

    def __len__(self) -> int:
        return len(self.sequences)

    # ---- per-sample augmentations ---------------------------------------

    def _time_resample(self, x: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = x.shape[0]
        if T <= 4 or np.random.rand() >= self.time_resample_prob:
            return x, missing
        lo, hi = self.time_resample_range
        rate = np.random.uniform(lo, hi)
        new_T = max(4, int(round(T * rate)))
        if new_T == T:
            return x, missing
        idx = np.linspace(0, T - 1, new_T).astype(np.int64)
        return x[idx], missing[idx]

    def _hflip(self, x: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() >= self.hflip_prob:
            return x, missing
        pts = x.reshape(x.shape[0], -1, 3).copy()
        pts[..., 0] = -pts[..., 0]

        lh = slice(_LEFT_HAND_LM_START, _LEFT_HAND_LM_START + _HAND_LEN)
        rh = slice(_RIGHT_HAND_LM_START, _RIGHT_HAND_LM_START + _HAND_LEN)
        pts[:, lh], pts[:, rh] = pts[:, rh].copy(), pts[:, lh].copy()
        missing = missing.copy()
        missing[:, lh], missing[:, rh] = missing[:, rh].copy(), missing[:, lh].copy()
        return pts.reshape(x.shape[0], -1), missing

    def _finger_dropout(self, x: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.finger_dropout_prob <= 0:
            return x, missing
        pts = x.reshape(x.shape[0], -1, 3).copy()
        missing = missing.copy()
        for hand_start in (_LEFT_HAND_LM_START, _RIGHT_HAND_LM_START):
            for finger in _FINGERS:
                if np.random.rand() < self.finger_dropout_prob:
                    lms = [hand_start + i for i in finger]
                    pts[:, lms] = 0.0
                    missing[:, lms] = True
        return pts.reshape(x.shape[0], -1), missing

    def _group_dropout(self, x: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.group_dropout_prob <= 0:
            return x, missing
        x = x.copy()
        missing = missing.copy()
        for start_lm, end_lm in _LANDMARK_GROUPS.values():
            if np.random.rand() < self.group_dropout_prob:
                f_start, f_end = start_lm * 3, end_lm * 3
                x[:, f_start:f_end] = 0.0
                missing[:, start_lm:end_lm] = True
        return x, missing

    def _affine(self, x: np.ndarray, missing: np.ndarray) -> np.ndarray:
        """Apply a 2D affine transform (rotation, scale, shear, translate) to
        x-y. z is scaled but not rotated/sheared.
        """
        if x.shape[0] == 0 or np.random.rand() >= self.affine_prob:
            return x
        lo, hi = self.affine_scale_range
        scale = np.random.uniform(lo, hi)
        shear_x = np.random.uniform(-self.affine_shear_max, self.affine_shear_max)
        shear_y = np.random.uniform(-self.affine_shear_max, self.affine_shear_max)
        tx = np.random.uniform(-self.affine_translate_max, self.affine_translate_max)
        ty = np.random.uniform(-self.affine_translate_max, self.affine_translate_max)
        theta = np.deg2rad(np.random.uniform(-self.affine_rotation_deg_max, self.affine_rotation_deg_max))

        c, s = np.cos(theta), np.sin(theta)
        # Compose order: rotation -> shear -> scale, then translate.
        rot   = np.array([[c, -s], [s, c]], dtype=np.float32)
        shear = np.array([[1.0, shear_x], [shear_y, 1.0]], dtype=np.float32)
        M = (rot @ shear) * scale

        pts = x.reshape(x.shape[0], -1, 3).astype(np.float32, copy=True)
        xy = pts[..., :2]
        xy = xy @ M.T
        xy[..., 0] += tx
        xy[..., 1] += ty
        pts[..., :2] = xy
        pts[..., 2] *= scale
        pts[missing] = 0.0
        return pts.reshape(x.shape[0], -1)

    def _time_mask(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] < 8 or np.random.rand() >= self.time_mask_prob:
            return x
        lo, hi = self.time_mask_frac_range
        span = max(1, int(round(np.random.uniform(lo, hi) * x.shape[0])))
        span = min(span, x.shape[0] - 1)
        start = np.random.randint(0, x.shape[0] - span + 1)
        x = x.copy()
        x[start : start + span] = 0.0
        return x

    def _spatial_mask(self, x: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pick a random axis-aligned rectangle in xy and zero any landmark
        whose mean position over the sequence falls inside it.
        """
        if x.shape[0] == 0 or np.random.rand() >= self.spatial_mask_prob:
            return x, missing
        lo, hi = self.spatial_mask_frac_range
        # The normalization layer brings most non-missing landmarks into
        # roughly [-1.5, 1.5]; sample mask in that range.
        bounds = 1.5
        frac_area = np.random.uniform(lo, hi)
        side = float(np.sqrt(frac_area)) * 2 * bounds
        cx = np.random.uniform(-bounds, bounds)
        cy = np.random.uniform(-bounds, bounds)
        x_lo, x_hi = cx - side / 2, cx + side / 2
        y_lo, y_hi = cy - side / 2, cy + side / 2

        pts = x.reshape(x.shape[0], -1, 3).copy()
        # Use mean per-landmark position across present frames so a passing
        # finger isn't half-masked. Missing frames don't contribute.
        present_count = (~missing).sum(axis=0).clip(min=1)
        mean_xy = (pts[..., :2] * (~missing)[..., None]).sum(axis=0) / present_count[..., None]
        in_box = (
            (mean_xy[:, 0] >= x_lo) & (mean_xy[:, 0] <= x_hi) &
            (mean_xy[:, 1] >= y_lo) & (mean_xy[:, 1] <= y_hi)
        )
        if in_box.any():
            pts[:, in_box] = 0.0
            missing = missing.copy()
            missing[:, in_box] = True
        return pts.reshape(x.shape[0], -1), missing

    # ---- main pipeline ---------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        row = self.sequences.iloc[idx]
        seq_id = int(row["sequence_id"])
        path = self.seq_dir / f"{seq_id}.npz"
        try:
            data = np.load(path)
            x = data["x"].copy()
            missing = data["missing"].copy().astype(bool)
            target = data["target"].astype(np.int64)
        except FileNotFoundError:
            return torch.zeros(0, N_FEATURES), torch.zeros(0, dtype=torch.long), 0, 0

        # Pre-normalization augmentations: time-domain and landmark-domain
        # changes that must happen before wrist-anchored normalization so
        # the normalization sees the "raw" sequence.
        if self.augment:
            x, missing = self._time_resample(x, missing)
            # CTC requires T >= L plus room for repeat-blanks. Resamples can
            # shrink T below L on rare long-target sequences.
            if x.shape[0] < len(target) + 4:
                # Re-load the unaugmented sequence instead of feeding CTC a
                # sample it can't align.
                data = np.load(path)
                x = data["x"].copy()
                missing = data["missing"].copy().astype(bool)
            x, missing = self._hflip(x, missing)
            x, missing = self._finger_dropout(x, missing)
            x, missing = self._group_dropout(x, missing)

        if self.max_frames is not None and x.shape[0] > self.max_frames:
            keep = np.linspace(0, x.shape[0] - 1, self.max_frames).astype(np.int64)
            x = x[keep]
            missing = missing[keep]

        x = normalize_sequence(x, missing)

        # Post-normalization augmentations: spatial transforms in normalized
        # space + masking. Affine before mask so the mask doesn't get warped.
        if self.augment:
            x = self._affine(x, missing)
            x, missing = self._spatial_mask(x, missing)
            x = self._time_mask(x)

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
