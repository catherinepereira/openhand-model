"""
One-shot preprocessing of the isolated signs competition data into
per-clip .npz files.

For each clip in train.csv:
  - Read its parquet shard
  - Pivot the (frame, row_id, x, y, z) long format into (T, 127, 3) using
    the canonical 127-landmark subset
  - Build the explicit missing mask (NaN -> True per landmark)
  - Save x + missing + sign label to a single .npz

Output layout:
  data/processed_signs/
    clips/<sequence_id>.npz   one file per clip (~5-50 KB each)
      x:       float32 (T, 127, 3)  NaN replaced with 0
      missing: bool    (T, 127)     True where landmark was absent
      label:   int32   ()           class index 0..249
    sign_to_idx.json              {"airplane": 0, "alligator": 1, ...}
    index.json                    {sequence_id: {participant, n_frames, label}}

Usage:
  python scripts/preprocess_signs.py
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
from model.signs_landmarks import N_LANDMARKS, ROW_ID_TO_LM_IDX  # noqa: E402

MODEL_ROOT = Path(__file__).resolve().parent.parent


def pivot_clip(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert a clip's long-format rows into (T, 127, 3) + (T, 127) missing.

    The Google ISLR parquet stores one row per (frame, type, landmark_index)
    with explicit `type` and `landmark_index` columns. We build a
    `<type>_<index>` key from those, look it up in ROW_ID_TO_LM_IDX, and
    write the (x, y, z) into the right slot. Rows outside our subset are
    discarded silently. NaN coordinates count as "missing".
    """
    # Build the "<type>_<index>" key vectorised. Faster than .apply.
    keys = df["type"].astype(str) + "_" + df["landmark_index"].astype(str)
    mask = keys.isin(ROW_ID_TO_LM_IDX)
    df = df[mask].copy()
    keys = keys[mask]
    if df.empty:
        return np.zeros((0, N_LANDMARKS, 3), dtype=np.float32), np.zeros((0, N_LANDMARKS), dtype=bool)

    frames = sorted(df["frame"].unique())
    frame_to_idx = {int(f): i for i, f in enumerate(frames)}
    T = len(frames)

    arr = np.full((T, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    # Map every kept row in one pass.
    frame_idx = df["frame"].map(frame_to_idx).to_numpy()
    lm_idx = keys.map(ROW_ID_TO_LM_IDX).to_numpy()
    xs = df["x"].to_numpy(dtype=np.float32)
    ys = df["y"].to_numpy(dtype=np.float32)
    zs = df["z"].to_numpy(dtype=np.float32)
    arr[frame_idx, lm_idx, 0] = xs
    arr[frame_idx, lm_idx, 1] = ys
    arr[frame_idx, lm_idx, 2] = zs

    missing = np.isnan(arr).any(axis=-1)
    x = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    return x, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=MODEL_ROOT / "data" / "raw_signs")
    ap.add_argument("--out", type=Path, default=MODEL_ROOT / "data" / "processed_signs")
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, process at most this many clips (debug)")
    args = ap.parse_args()

    clips_dir = args.out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.raw / "train.csv"
    if not train_csv.exists():
        sys.exit(f"train.csv not found at {train_csv}. Run download_signs_data.py first.")

    sign_map_path = args.raw / "sign_to_prediction_index_map.json"
    if not sign_map_path.exists():
        sys.exit(f"sign_to_prediction_index_map.json not found at {sign_map_path}")
    with open(sign_map_path) as f:
        sign_to_idx: dict[str, int] = json.load(f)

    meta = pd.read_csv(train_csv)
    if args.limit:
        meta = meta.sample(n=min(args.limit, len(meta)), random_state=42).reset_index(drop=True)
    print(f"Processing {len(meta)} clips (one parquet per clip)")

    index: dict[str, dict] = {}

    for row in tqdm(meta.itertuples(index=False), total=len(meta)):
        path = row.path
        seq_id = int(row.sequence_id)
        sign = str(row.sign)
        label = sign_to_idx[sign]
        participant = int(row.participant_id) if hasattr(row, "participant_id") else -1

        out_npz = clips_dir / f"{seq_id}.npz"
        if out_npz.exists():
            # Resumable: skip clips already written.
            index[str(seq_id)] = {
                "path": path,
                "participant_id": participant,
                "n_frames": -1,
                "sign": sign,
                "label": label,
            }
            continue

        parquet = args.raw / path
        try:
            clip_df = pd.read_parquet(parquet)
        except FileNotFoundError:
            print(f"  parquet missing: {parquet}, skipping {seq_id}")
            continue

        x, missing = pivot_clip(clip_df)
        if x.shape[0] == 0:
            continue

        np.savez_compressed(out_npz, x=x, missing=missing, label=np.int32(label))
        index[str(seq_id)] = {
            "path": path,
            "participant_id": participant,
            "n_frames": int(x.shape[0]),
            "sign": sign,
            "label": label,
        }

    with open(args.out / "sign_to_idx.json", "w") as f:
        json.dump(sign_to_idx, f, indent=2)
    with open(args.out / "index.json", "w") as f:
        json.dump(index, f)

    print(f"\nSaved {len(index)} clips to {clips_dir}/")
    total_mb = sum(p.stat().st_size for p in clips_dir.glob("*.npz")) / 1e6
    print(f"Total size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
