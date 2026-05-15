"""
Landmark selection for the fingerspelling CTC model.

127 selected MediaPipe Holistic landmarks per frame (40 lips + 16 left-eye +
16 right-eye + 4 nose + 9 pose + 2 * 21 hands), 3 axes each, 381 floats per
frame total.

Module exports:
- SELECTED_COLS: list[str] of parquet column names in canonical order
- N_LANDMARKS, N_FEATURES: convenience constants
- normalize_sequence(arr): per-sequence wrist-anchored normalization

MUST stay in sync with openhand/backend/services/ctc_landmarks.py. That file
is a deliberate duplicate (the backend doesn't import from this repo). Any
change to the index lists, group order, or normalization formula here must
be mirrored there, or the deployed CTC model will see inputs with the wrong
shape/ordering and silently produce garbage.
"""

from __future__ import annotations

import numpy as np

# MediaPipe FaceMesh indices for the lips outline (40 points).
LIPS_IDX = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
# Left and right eye outlines (16 each).
LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
NOSE_IDX = [1, 2, 98, 327]

# MediaPipe Pose indices: 0=nose, 11/12=shoulders, 13/14=elbows, 15/16=wrists, 23/24=hips.
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]

HAND_IDX = list(range(21))


def _face_cols(prefix: str, indices: list[int]) -> list[str]:
    return [f"{ax}_{prefix}_{i}" for i in indices for ax in ("x", "y", "z")]

def _build_columns() -> tuple[list[str], dict[str, int]]:
    cols: list[str] = []
    face_indices = LIPS_IDX + LEFT_EYE_IDX + RIGHT_EYE_IDX + NOSE_IDX
    cols += _face_cols("face", face_indices)
    cols += [f"{ax}_pose_{i}" for i in POSE_IDX for ax in ("x", "y", "z")]
    cols += [f"{ax}_left_hand_{i}"  for i in HAND_IDX for ax in ("x", "y", "z")]
    cols += [f"{ax}_right_hand_{i}" for i in HAND_IDX for ax in ("x", "y", "z")]

    group_offsets = {
        "face":       0,
        "pose":       len(face_indices) * 3,
        "left_hand":  (len(face_indices) + len(POSE_IDX)) * 3,
        "right_hand": (len(face_indices) + len(POSE_IDX) + len(HAND_IDX)) * 3,
    }
    return cols, group_offsets


SELECTED_COLS, GROUP_OFFSETS = _build_columns()

N_LANDMARKS = (
    len(LIPS_IDX) + len(LEFT_EYE_IDX) + len(RIGHT_EYE_IDX) + len(NOSE_IDX)
    + len(POSE_IDX) + 2 * len(HAND_IDX)
)
N_FEATURES = N_LANDMARKS * 3

assert N_FEATURES == len(SELECTED_COLS), (N_FEATURES, len(SELECTED_COLS))


def normalize_sequence(arr: np.ndarray, missing: np.ndarray | None = None) -> np.ndarray:
    """
    Normalize a (T, N_FEATURES) sequence and return float32.

    Strategy:
      1. Anchor around the mean of the dominant-hand wrist position across
         the sequence.
      2. Divide by a robust scale (95th-percentile |value| across kept
         landmarks).
      3. Re-zero originally-missing landmarks so the model can detect them.

    missing is a (T, N_LANDMARKS) bool mask, True where a landmark was
    originally absent. If None, we treat all-zero (x,y,z) triplets as missing.
    """
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32, copy=True)
    pts = arr.reshape(arr.shape[0], -1, 3)

    if missing is None:
        missing = (pts == 0).all(axis=-1)
    missing = missing.astype(bool, copy=False)

    rh_lm = GROUP_OFFSETS["right_hand"] // 3
    lh_lm = GROUP_OFFSETS["left_hand"] // 3
    right_present_frames = ~missing[:, rh_lm]
    left_present_frames  = ~missing[:, lh_lm]

    if right_present_frames.sum() >= left_present_frames.sum():
        wrist_xyz = pts[:, rh_lm, :]
        wrist_mask = right_present_frames
    else:
        wrist_xyz = pts[:, lh_lm, :]
        wrist_mask = left_present_frames

    if wrist_mask.any():
        anchor = wrist_xyz[wrist_mask].mean(axis=0)
    else:
        anchor = np.zeros(3, dtype=np.float32)

    pts = pts - anchor

    present_pts = pts[~missing]
    if present_pts.size:
        scale = np.percentile(np.abs(present_pts), 95)
        if scale > 1e-6:
            pts = pts / scale

    pts[missing] = 0.0
    return pts.reshape(arr.shape[0], -1).astype(np.float32)
