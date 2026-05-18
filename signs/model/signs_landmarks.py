"""
Landmark selection + engineered features for the isolated sign classifier.

The Kaggle "Google Isolated Sign Language Recognition" data ships
MediaPipe Holistic landmarks (543 total per frame: 468 face + 33 pose +
21 left hand + 21 right hand). We subset to the same 127 landmarks the
CTC fingerspelling pipeline uses, then add engineered motion + spatial
features per the competition winners' patterns (mean/std normalization,
motion deltas between adjacent frames, hand-to-lip pairwise distances).

Frame layout in the parquet shards:
  Each row is (frame, row_id, x, y, z) where row_id encodes
  "<type>_<index>" e.g. "face_61" or "right_hand_5".

Output:
  N_LANDMARKS = 127
  N_BASE_FEATURES = 127 * 3 = 381
  N_AUG_FEATURES = motion (381) + hand-lip distances (42) = 423
  N_FEATURES = 381 + 423 = 804

This module is the *training* side. The deployed backend has a parallel
copy (openhand/backend/services/signs_landmarks.py) that must stay in
sync; a parity test catches drift.
"""

from __future__ import annotations

import numpy as np

# MediaPipe Holistic landmark indices (matches the CTC pipeline's
# landmarks.py). Same physical points, just used in a different context.
LIPS_IDX = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
NOSE_IDX = [1, 2, 98, 327]
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
HAND_IDX = list(range(21))

N_FACE_LM = len(LIPS_IDX) + len(LEFT_EYE_IDX) + len(RIGHT_EYE_IDX) + len(NOSE_IDX)  # 76
N_POSE_LM = len(POSE_IDX)                                                            # 9
N_HAND_LM = len(HAND_IDX)                                                            # 21
N_LANDMARKS = N_FACE_LM + N_POSE_LM + 2 * N_HAND_LM                                  # 127
N_COORDS = 3
N_BASE_FEATURES = N_LANDMARKS * N_COORDS                                             # 381

# Per-group slice offsets (in landmark units, not features).
GROUP_OFFSETS: dict[str, int] = {}
_cursor = 0
GROUP_OFFSETS["face"] = _cursor
_cursor += N_FACE_LM
GROUP_OFFSETS["pose"] = _cursor
_cursor += N_POSE_LM
GROUP_OFFSETS["left_hand"] = _cursor
_cursor += N_HAND_LM
GROUP_OFFSETS["right_hand"] = _cursor

# Useful sub-slices inside the 127-landmark vector.
LIPS_SLICE = slice(0, len(LIPS_IDX))  # 40 landmarks
LEFT_HAND_SLICE = slice(GROUP_OFFSETS["left_hand"], GROUP_OFFSETS["left_hand"] + N_HAND_LM)
RIGHT_HAND_SLICE = slice(GROUP_OFFSETS["right_hand"], GROUP_OFFSETS["right_hand"] + N_HAND_LM)


def _parquet_row_id_to_landmark_idx() -> dict[str, int]:
    """Build a flat lookup: parquet row_id string -> index into the
    127-landmark output vector. Used by preprocess_signs.py to pivot the
    long-format parquet rows into a (T, 127, 3) tensor."""
    out: dict[str, int] = {}
    cursor = 0
    # Face landmarks (combined into one "face" group order)
    face_indices = LIPS_IDX + LEFT_EYE_IDX + RIGHT_EYE_IDX + NOSE_IDX
    for raw_idx in face_indices:
        out[f"face_{raw_idx}"] = cursor
        cursor += 1
    for raw_idx in POSE_IDX:
        out[f"pose_{raw_idx}"] = cursor
        cursor += 1
    for raw_idx in HAND_IDX:
        out[f"left_hand_{raw_idx}"] = cursor
        cursor += 1
    for raw_idx in HAND_IDX:
        out[f"right_hand_{raw_idx}"] = cursor
        cursor += 1
    assert cursor == N_LANDMARKS, (cursor, N_LANDMARKS)
    return out


ROW_ID_TO_LM_IDX = _parquet_row_id_to_landmark_idx()


# Hand-to-lip distances: for each of the 21 hand landmarks, distance to
# the centroid of the lips. Computed per hand (left + right). 21 * 2 = 42
# distances per frame. We use a single centroid rather than 40 individual
# lip-landmark pairs to keep the feature count manageable.
N_HAND_LIP_DISTANCES = 2 * N_HAND_LM  # 42

# Motion deltas: per-frame difference of every (x, y, z), padded so the
# output has the same T as the input. 381 features per frame.
N_MOTION_FEATURES = N_BASE_FEATURES  # 381

N_AUG_FEATURES = N_MOTION_FEATURES + N_HAND_LIP_DISTANCES  # 423
N_FEATURES = N_BASE_FEATURES + N_AUG_FEATURES              # 804


def normalize_clip(arr: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Per-clip mean/std normalization (Kaggle winners' standard).

    arr:     (T, 127, 3) float32. NaNs already replaced with 0 by caller.
    missing: (T, 127) bool. True where landmark was absent.

    Returns (T, 127, 3) float32. Missing landmarks are zeroed *after*
    normalization so the model can still detect absence.
    """
    arr = arr.copy().astype(np.float32, copy=False)
    # Compute mean/std over present landmarks only, per coord axis.
    present_mask = ~missing  # (T, 127)
    flat = arr.reshape(-1, 3)
    flat_mask = present_mask.reshape(-1)
    present = flat[flat_mask]
    if present.size:
        mean = present.mean(axis=0)
        std = present.std(axis=0)
        std = np.maximum(std, 1e-6)
    else:
        mean = np.zeros(3, dtype=np.float32)
        std = np.ones(3, dtype=np.float32)
    arr = (arr - mean) / std
    # Re-zero missing landmarks after normalization (the contract: zeros
    # mean "absent," combined with the missing mask passed to the model).
    arr[missing] = 0.0
    return arr.astype(np.float32, copy=False)


def build_features(
    arr: np.ndarray,
    missing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack (T, 127, 3) into (T, N_FEATURES) plus the (T, 127) missing
    mask passed through unchanged.

    Features per frame:
      0..380  : flattened normalized landmarks (127 * 3)
      381..761: motion deltas (frame[t] - frame[t-1]); first frame is 0
      762..803: hand-to-lip centroid distances (42 floats)

    arr must already be normalized (see normalize_clip).
    """
    T = arr.shape[0]
    if T == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32), missing

    base = arr.reshape(T, N_BASE_FEATURES)

    # Motion deltas
    motion = np.zeros_like(base)
    if T > 1:
        motion[1:] = base[1:] - base[:-1]

    # Hand-to-lip distances
    lip_centroid = arr[:, LIPS_SLICE, :].mean(axis=1)  # (T, 3)
    left_hand = arr[:, LEFT_HAND_SLICE, :]              # (T, 21, 3)
    right_hand = arr[:, RIGHT_HAND_SLICE, :]            # (T, 21, 3)
    left_dist = np.linalg.norm(
        left_hand - lip_centroid[:, None, :], axis=-1,
    ).astype(np.float32)  # (T, 21)
    right_dist = np.linalg.norm(
        right_hand - lip_centroid[:, None, :], axis=-1,
    ).astype(np.float32)  # (T, 21)
    hand_lip = np.concatenate([left_dist, right_dist], axis=1)  # (T, 42)

    features = np.concatenate([base, motion, hand_lip], axis=1).astype(np.float32)
    assert features.shape == (T, N_FEATURES), (features.shape, T, N_FEATURES)
    return features, missing
