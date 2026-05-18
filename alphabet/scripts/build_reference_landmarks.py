"""
Pick per-letter medoid landmark vectors from the preprocessed alphabet
dataset and write them as a single JSON artifact for the OpenHand
"Learn the signs" screen.

For each letter, the medoid is the real training sample whose 63-float
landmark vector is closest (Euclidean) to that letter's class centroid.
Mirrors the medoid-clip approach used for the isolated-sign references:
a real pose, not an averaged ghost.

Run after preprocess_alphabet.py has produced X.npy/y.npy. Output is
small (~10KB) and gets copied into the openhand backend so the Learn
screen has no runtime dependency on this repo or the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODEL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = MODEL_ROOT / "data" / "processed_alphabet"
DEFAULT_OUT = MODEL_ROOT / "exports" / "reference_landmarks.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="Directory with X.npy, y.npy, label_map.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    X = np.load(args.data / "X.npy")
    y = np.load(args.data / "y.npy")
    with open(args.data / "label_map.json") as f:
        label_map: dict[str, str] = json.load(f)

    if X.shape[1] != 63:
        raise SystemExit(f"Expected (N, 63) landmarks, got {X.shape}")

    per_letter: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for idx_str, letter in label_map.items():
        idx = int(idx_str)
        mask = y == idx
        n = int(mask.sum())
        if n == 0:
            print(f"  {letter}: no samples, skipping")
            continue
        samples = X[mask]
        centroid = samples.mean(axis=0)
        dists = np.linalg.norm(samples - centroid, axis=1)
        medoid = samples[int(np.argmin(dists))].astype(np.float32)
        per_letter[letter.upper()] = medoid.tolist()
        counts[letter.upper()] = n

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "21 landmarks * (x, y, z), wrist-anchored, p95-scaled; per-letter medoid sample",
        "n_features": 63,
        "letters": per_letter,
        "sample_counts": counts,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)

    total_letters = len(per_letter)
    total_samples = sum(counts.values())
    print(f"Wrote {args.out}")
    print(f"  {total_letters} letters, medoid selected from {total_samples} total samples")


if __name__ == "__main__":
    main()
