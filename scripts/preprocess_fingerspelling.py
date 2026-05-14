"""
One-time pre-extraction of fingerspelling sequences to fast .npz files.

For each sequence in train.csv:
  - Load its frames from the appropriate Parquet shard (touching each shard
    exactly once across the whole dataset)
  - Select the 127-landmark Kaggle-winner subset (381 features/frame)
  - Save a single .npz with the landmarks array AND an explicit missing mask
    (so downstream training does NOT have to use 0-as-sentinel)

Output layout:
  data/processed_fingerspelling/
    sequences/<sequence_id>.npz   # one file per sequence (small, ~10-100 KB)
      x:       float32 (T, N_FEATURES)  — NaN-filled with 0
      missing: bool   (T, N_LANDMARKS)  — True where landmark was originally NaN
      target:  int16  (L,)              — character indices
    index.json  # quick lookup: sequence_id -> {target, n_frames}

Usage:
  python scripts/preprocess_fingerspelling.py
  python scripts/preprocess_fingerspelling.py --limit 16000  # subset, faster
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.landmarks import SELECTED_COLS, N_LANDMARKS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", type=Path)
    ap.add_argument("--out", default="data/processed_fingerspelling", type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, process at most this many sequences (with same "
                         "random seed used by --subset in train_ctc.py)")
    args = ap.parse_args()

    seq_dir = args.out / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    with open(args.raw / "character_to_prediction_index.json") as f:
        char_to_idx = json.load(f)

    meta = pd.read_csv(args.raw / "train.csv")
    meta = meta[meta["phrase"].str.len() > 0]
    if args.limit:
        meta = meta.sample(n=min(args.limit, len(meta)), random_state=42).reset_index(drop=True)
    # Sort by file_id so we touch each Parquet shard exactly once
    meta = meta.sort_values("file_id").reset_index(drop=True)

    print(f"Processing {len(meta)} sequences from {meta['file_id'].nunique()} shards")

    index: dict[str, dict] = {}
    current_fid = None
    current_df = None

    for row in tqdm(meta.itertuples(index=False), total=len(meta)):
        file_id = int(row.file_id)
        seq_id = int(row.sequence_id)
        phrase = str(row.phrase).lower()

        if file_id != current_fid:
            shard = args.raw / "train_landmarks" / f"{file_id}.parquet"
            current_df = pd.read_parquet(shard, columns=SELECTED_COLS)
            current_fid = file_id

        try:
            frames = current_df.loc[seq_id]
            if isinstance(frames, pd.Series):
                frames = frames.to_frame().T
            arr = frames.values.astype(np.float32)
        except KeyError:
            continue
        if arr.shape[0] == 0:
            continue

        # Per-landmark missing mask: a landmark counts as missing when all 3
        # of its (x, y, z) values were NaN.
        per_lm = arr.reshape(arr.shape[0], N_LANDMARKS, 3)
        missing = np.isnan(per_lm).all(axis=-1)  # (T, N_LANDMARKS)

        # Fill NaNs with 0 for the saved array (downstream code will use the
        # explicit `missing` mask, not zero-equality).
        x = np.nan_to_num(arr, nan=0.0).astype(np.float32)

        target_ids = np.array(
            [char_to_idx[c] for c in phrase if c in char_to_idx],
            dtype=np.int16,
        )
        if target_ids.size == 0:
            continue

        np.savez_compressed(
            seq_dir / f"{seq_id}.npz",
            x=x,
            missing=missing,
            target=target_ids,
        )
        index[str(seq_id)] = {
            "file_id": file_id,
            "n_frames": int(x.shape[0]),
            "target_len": int(target_ids.size),
            "participant_id": int(row.participant_id),
        }

    with open(args.out / "index.json", "w") as f:
        json.dump(index, f)

    print(f"\nSaved {len(index)} sequences to {seq_dir}/")
    total_mb = sum(p.stat().st_size for p in seq_dir.glob("*.npz")) / 1e6
    print(f"Total size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
