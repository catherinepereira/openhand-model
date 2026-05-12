"""
Preprocess the ASL Fingerspelling Parquet files into a compact NumPy dataset.

Dataset structure:
  - data/raw/train_landmarks/<file_id>.parquet  — landmark frames, indexed by sequence_id
  - data/raw/train.csv                          — maps sequence_id → phrase + file_id
  - data/raw/character_to_prediction_index.json — char → int label

Each sequence is one person fingerspelling a multi-character phrase.
We split each sequence's frames evenly across characters and take the
median frame per character segment as a static 63-float feature vector.

Column naming in parquets: x_right_hand_0..20, y_right_hand_0..20, z_right_hand_0..20
(same for left_hand). We prefer right hand, fall back to left when right is all-NaN.

Output (data/processed/):
  X.npy            float32  (N, 63)
  y.npy            int64    (N,)
  label_map.json   {index: char}

Usage:
  python scripts/preprocess.py [--raw data/raw] [--out data/processed] [--workers 4]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def _hand_cols(side: str) -> list[str]:
    """Return the 63 column names for one hand in x,y,z interleaved order."""
    cols = []
    for i in range(21):
        cols.append(f"x_{side}_hand_{i}")
        cols.append(f"y_{side}_hand_{i}")
        cols.append(f"z_{side}_hand_{i}")
    return cols


RIGHT_COLS = _hand_cols("right")
LEFT_COLS  = _hand_cols("left")


def extract_median_frame(frames: pd.DataFrame, cols: list[str]) -> np.ndarray | None:
    """Extract and median-pool available frames for one hand. Returns (63,) or None."""
    avail = [c for c in cols if c in frames.columns]
    if not avail:
        return None
    sub = frames[avail].copy()
    # Drop rows where this hand is entirely NaN
    sub = sub.dropna(how="all")
    if sub.empty:
        return None
    # Fill any remaining NaNs with 0, pad missing cols with 0
    for c in cols:
        if c not in sub.columns:
            sub[c] = 0.0
    sub = sub[cols].fillna(0.0)
    return sub.median(axis=0).values.astype(np.float32)


def normalize(vec: np.ndarray) -> np.ndarray:
    """Wrist-centre and unit-scale the landmark vector."""
    vec = vec.copy()
    # Wrist = landmark 0: cols 0,1,2 = x0,y0,z0
    wrist = vec[:3].copy()
    for i in range(21):
        vec[i*3 : i*3+3] -= wrist
    scale = np.percentile(np.abs(vec), 95)
    if scale > 1e-6:
        vec /= scale
    return vec


def process_sequence(
    frames: pd.DataFrame,
    phrase: str,
    char_to_idx: dict[str, int],
) -> tuple[list[np.ndarray], list[int]]:
    """
    Split a sequence's frames evenly across characters in the phrase,
    extract one median landmark vector per character.
    """
    n_frames = len(frames)
    n_chars  = len(phrase)
    if n_frames == 0 or n_chars == 0:
        return [], []

    # Divide frames into n_chars roughly equal segments
    splits = np.array_split(np.arange(n_frames), n_chars)

    xs, ys = [], []
    for char, idx_range in zip(phrase, splits):
        if char not in char_to_idx:
            continue
        label = char_to_idx[char]
        segment = frames.iloc[idx_range]

        vec = extract_median_frame(segment, RIGHT_COLS)
        if vec is None:
            vec = extract_median_frame(segment, LEFT_COLS)
        if vec is None:
            continue

        vec = normalize(vec)
        xs.append(vec)
        ys.append(label)

    return xs, ys


def process_parquet(
    parquet_path: Path,
    meta_subset: pd.DataFrame,
    char_to_idx: dict[str, int],
) -> tuple[list, list]:
    """Process one parquet file, returning (features, labels) for all sequences."""
    try:
        # Only load hand columns + index to save memory
        hand_cols = (
            [f"x_right_hand_{i}" for i in range(21)] +
            [f"y_right_hand_{i}" for i in range(21)] +
            [f"z_right_hand_{i}" for i in range(21)] +
            [f"x_left_hand_{i}"  for i in range(21)] +
            [f"y_left_hand_{i}"  for i in range(21)] +
            [f"z_left_hand_{i}"  for i in range(21)]
        )
        df = pd.read_parquet(parquet_path, columns=hand_cols)
    except Exception as e:
        print(f"  Warning: could not read {parquet_path.name}: {e}")
        return [], []

    all_x, all_y = [], []
    for _, row in meta_subset.iterrows():
        seq_id = row["sequence_id"]
        phrase = str(row["phrase"]).lower()
        try:
            frames = df.loc[seq_id]
            # loc returns a Series if only one row — convert back to DataFrame
            if isinstance(frames, pd.Series):
                frames = frames.to_frame().T
        except KeyError:
            continue

        xs, ys = process_sequence(frames, phrase, char_to_idx)
        all_x.extend(xs)
        all_y.extend(ys)

    return all_x, all_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",  default="data/raw",       type=Path)
    parser.add_argument("--out",  default="data/processed", type=Path)
    parser.add_argument("--limit", default=None, type=int,
                        help="Limit number of parquet files (for quick testing)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load char map
    with open(args.raw / "character_to_prediction_index.json") as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    print(f"Characters: {len(char_to_idx)}  (includes letters, digits, punctuation, space)")

    # Load metadata
    meta = pd.read_csv(args.raw / "train.csv")
    print(f"Sequences: {len(meta)}")

    # Find parquet files
    parquet_files = sorted((args.raw / "train_landmarks").glob("*.parquet"))
    if args.limit:
        parquet_files = parquet_files[:args.limit]
    print(f"Parquet shards to process: {len(parquet_files)}")

    all_x, all_y = [], []
    for pf in tqdm(parquet_files, desc="Shards"):
        file_id = int(pf.stem)
        meta_sub = meta[meta["file_id"] == file_id]
        if meta_sub.empty:
            continue
        xs, ys = process_parquet(pf, meta_sub, char_to_idx)
        all_x.extend(xs)
        all_y.extend(ys)

    if not all_x:
        print("No samples extracted. Check data paths.")
        return

    X = np.stack(all_x).astype(np.float32)
    y = np.array(all_y, dtype=np.int64)

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Classes present: {len(np.unique(y))} / {len(char_to_idx)}")

    # Per-class counts for letters a-z
    print("\nLetter distribution:")
    for char in "abcdefghijklmnopqrstuvwxyz":
        if char in char_to_idx:
            idx = char_to_idx[char]
            count = int((y == idx).sum())
            print(f"  {char.upper()}: {count}")

    np.save(args.out / "X.npy", X)
    np.save(args.out / "y.npy", y)
    with open(args.out / "label_map.json", "w") as f:
        json.dump(idx_to_char, f, indent=2)

    print(f"\nSaved to {args.out}/")
    print("Next: python scripts/train.py")


if __name__ == "__main__":
    main()
