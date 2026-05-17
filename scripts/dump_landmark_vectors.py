"""
Dump deterministic Python landmark-pipeline outputs to JSON so the
TypeScript port can be compared element-wise against them.

Run once when the Python landmark code or the canonical group order
changes; commit the resulting JSON next to the test that consumes it.

Produces ``frontend/src/lib/__tests__/landmark_fixtures.json`` with:

- raw_frame: synthetic per-group landmark inputs (small deterministic
  numbers, one frame). Same shape the TS RawFrameLandmarks expects.
- packed_features: the (N_FEATURES,) feature vector Python's
  build_frame_features produces from that input.
- packed_missing: the (N_LANDMARKS,) missing mask Python emits.

The TS test loads the JSON and asserts
buildFrameFeatures(raw_frame) ~= {packed_features, packed_missing}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.landmarks import (  # noqa: E402
    LIPS_IDX, LEFT_EYE_IDX, RIGHT_EYE_IDX, NOSE_IDX, POSE_IDX, HAND_IDX,
    N_LANDMARKS, N_FEATURES,
)

# MediaPipe's max indices for each source. We need to provide a
# landmark per *raw* index so the indexing into the source lists works.
FACE_MAX_IDX = max(max(LIPS_IDX), max(LEFT_EYE_IDX), max(RIGHT_EYE_IDX), max(NOSE_IDX)) + 1
POSE_MAX_IDX = max(POSE_IDX) + 1
HAND_MAX_IDX = max(HAND_IDX) + 1


def synth_landmark(i: int, scale: float) -> dict[str, float]:
    """Make a deterministic, distinguishable landmark for index i."""
    return {
        "x": (i * 0.001 + scale) % 1.0,
        "y": (i * 0.002 + scale * 0.5) % 1.0,
        "z": (i * 0.003 - scale * 0.25) % 1.0,
    }


def build_raw_frame():
    """The synthetic input both pipelines will consume."""
    return {
        "face": [synth_landmark(i, 0.10) for i in range(FACE_MAX_IDX)],
        "pose": [synth_landmark(i, 0.30) for i in range(POSE_MAX_IDX)],
        "leftHand": [synth_landmark(i, 0.55) for i in range(HAND_MAX_IDX)],
        "rightHand": [synth_landmark(i, 0.77) for i in range(HAND_MAX_IDX)],
    }


def pack_frame_python(raw_frame) -> tuple[np.ndarray, np.ndarray]:
    """Replicate build_frame_features inline so the test fixture is
    completely self-contained (the backend mirror has its own copy).
    Both must match this exactly."""
    features = np.zeros(N_FEATURES, dtype=np.float32)
    missing = np.ones(N_LANDMARKS, dtype=bool)
    cursor = 0

    def write(group_lms, indices):
        nonlocal cursor
        for i in indices:
            point = group_lms[i]
            base = cursor * 3
            features[base] = point["x"]
            features[base + 1] = point["y"]
            features[base + 2] = point["z"]
            missing[cursor] = False
            cursor += 1

    write(raw_frame["face"], LIPS_IDX)
    write(raw_frame["face"], LEFT_EYE_IDX)
    write(raw_frame["face"], RIGHT_EYE_IDX)
    write(raw_frame["face"], NOSE_IDX)
    write(raw_frame["pose"], POSE_IDX)
    write(raw_frame["leftHand"], HAND_IDX)
    write(raw_frame["rightHand"], HAND_IDX)
    return features, missing


def main():
    out = Path(__file__).parent.parent.parent / "openhand" / "frontend" / "src" / "lib" / "__tests__" / "landmark_fixtures.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    raw_frame = build_raw_frame()
    packed_features, packed_missing = pack_frame_python(raw_frame)

    fixtures = {
        "metadata": {
            "n_landmarks": int(N_LANDMARKS),
            "n_features": int(N_FEATURES),
        },
        "raw_frame": raw_frame,
        "packed_features": packed_features.tolist(),
        "packed_missing": packed_missing.tolist(),
    }

    with open(out, "w") as f:
        json.dump(fixtures, f)
    print(f"Wrote {out}")
    print(f"  N_LANDMARKS={N_LANDMARKS}, N_FEATURES={N_FEATURES}")
    print(f"  packed_features len={len(fixtures['packed_features'])}")


if __name__ == "__main__":
    main()
