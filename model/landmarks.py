"""
Landmark selection for the fingerspelling CTC model.

We follow the Kaggle 1st-place approach: 130 selected MediaPipe Holistic
landmarks per frame, drawn from face/lips/eyes/pose/hands. 3 axes each → 390
floats per frame.

Module exports:
- SELECTED_COLS: list[str] of parquet column names in canonical order
- N_LANDMARKS, N_FEATURES: convenience constants
- normalize_sequence(arr): per-sequence wrist-anchored normalisation
"""

from __future__ import annotations

import numpy as np

# MediaPipe FaceMesh indices for the lips outline (40 points) and selected
# eye/nose-bridge anchors. These are the standard "interesting" face points
# referenced by most ASL fingerspelling Kaggle solutions.
LIPS_IDX = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
# Left + right eye outlines (16 each = 32). Kept compact — most variance is in
# the lips/hands; eyes give a head-orientation anchor only.
LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Nose bridge: 4 points
NOSE_IDX = [1, 2, 98, 327]

# MediaPipe Pose indices to keep — upper body only.
# 11=L shoulder, 12=R shoulder, 13=L elbow, 14=R elbow, 15=L wrist, 16=R wrist,
# 23=L hip, 24=R hip, 0=nose.
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]

# Hand indices: all 21 from each hand.
HAND_IDX = list(range(21))

# Build the canonical ordered column list. Order matters because the model
# consumes a fixed-shape (T, N_FEATURES) tensor.
def _face_cols(prefix: str, indices: list[int]) -> list[str]:
    return [f"{ax}_{prefix}_{i}" for i in indices for ax in ("x", "y", "z")]

def _build_columns() -> tuple[list[str], dict[str, int]]:
    cols: list[str] = []
    # Face landmark groups (in order: lips, left eye, right eye, nose)
    face_indices = LIPS_IDX + LEFT_EYE_IDX + RIGHT_EYE_IDX + NOSE_IDX
    cols += _face_cols("face", face_indices)
    # Pose
    cols += [f"{ax}_pose_{i}" for i in POSE_IDX for ax in ("x", "y", "z")]
    # Hands
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
N_FEATURES = N_LANDMARKS * 3  # x, y, z per landmark

assert N_FEATURES == len(SELECTED_COLS), (N_FEATURES, len(SELECTED_COLS))


def normalize_sequence(arr: np.ndarray, missing: np.ndarray | None = None) -> np.ndarray:
    """
    Normalise a (T, N_FEATURES) sequence and return float32.

    Strategy (per-sequence, batch-time invariant):
      1. Anchor around the mean of the dominant-hand wrist position across the
         sequence (translation invariance — captures the signer's rest pose).
      2. Divide by a robust scale (95th-percentile |value| across kept
         landmarks).
      3. Re-zero originally-missing landmarks so the model can detect them.

    ``missing`` is a (T, N_LANDMARKS) bool mask — True where a landmark was
    originally absent. If None, we treat all-zero (x,y,z) triplets as missing
    (legacy path, kept for back-compat with code that didn't track NaN
    explicitly).
    """
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32, copy=True)
    pts = arr.reshape(arr.shape[0], -1, 3)  # (T, N_LANDMARKS, 3)

    if missing is None:
        missing = (pts == 0).all(axis=-1)
    missing = missing.astype(bool, copy=False)

    # Dominant-hand wrist anchor: whichever wrist is present in more frames.
    rh_lm = GROUP_OFFSETS["right_hand"] // 3  # landmark index of right wrist
    lh_lm = GROUP_OFFSETS["left_hand"] // 3
    right_present_frames = ~missing[:, rh_lm]   # (T,)
    left_present_frames  = ~missing[:, lh_lm]

    if right_present_frames.sum() >= left_present_frames.sum():
        wrist_xyz = pts[:, rh_lm, :]            # (T, 3)
        wrist_mask = right_present_frames
    else:
        wrist_xyz = pts[:, lh_lm, :]
        wrist_mask = left_present_frames

    if wrist_mask.any():
        anchor = wrist_xyz[wrist_mask].mean(axis=0)
    else:
        anchor = np.zeros(3, dtype=np.float32)

    # Translate
    pts = pts - anchor

    # Robust scale over present landmarks only
    present_pts = pts[~missing]
    if present_pts.size:
        scale = np.percentile(np.abs(present_pts), 95)
        if scale > 1e-6:
            pts = pts / scale

    # Re-zero missing landmarks AFTER scaling, so they remain an unambiguous
    # "this point was absent" signal.
    pts[missing] = 0.0
    return pts.reshape(arr.shape[0], -1).astype(np.float32)
