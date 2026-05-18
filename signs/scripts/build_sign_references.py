"""
Build per-sign medoid-clip references for the Learn-the-words screen.

For each of 250 signs, run every training clip through the trained
encoder, pool to a (d_model,) embedding, take the class centroid, and
pick the real clip closest to that centroid. The medoid clip is the
"most prototypical" example of the sign according to the model itself.

Output:
  exports/signs/sign_references.json
    {
      "sign_to_idx": {...},
      "signs": {
        "airplane": {
          "label": 0,
          "n_frames": 47,
          "landmarks": [ ... flat T*127*3 floats ... ],
          "missing":   [ ... flat T*127 bools ... ]
        },
        ...
      }
    }

The frontend animates each medoid clip on the HandModel3D and uses it as
the reference for grading the user's attempt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.sign_classifier import build_sign_classifier, encode_features  # noqa: E402
from model.signs_dataset import (  # noqa: E402
    SignsDataset,
    load_sign_map,
    signs_collate,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=MODEL_ROOT / "data" / "processed_signs", type=Path)
    ap.add_argument("--exports", default=MODEL_ROOT / "exports", type=Path)
    ap.add_argument("--checkpoint", default=MODEL_ROOT / "exports" / "best.pt", type=Path)
    ap.add_argument("--meta", default=MODEL_ROOT / "exports" / "model_meta.json", type=Path)
    ap.add_argument("--batch", default=64, type=int)
    ap.add_argument("--workers", default=2, type=int)
    ap.add_argument("--max_frames", default=80, type=int)
    args = ap.parse_args()

    sign_to_idx, idx_to_sign = load_sign_map(args.processed)
    num_classes = len(sign_to_idx)

    with open(args.meta) as f:
        meta = json.load(f)

    model = build_sign_classifier(
        num_classes=num_classes,
        d_model=meta["d_model"],
        nhead=meta["nhead"],
        num_layers=meta["layers"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()

    with open(args.processed / "index.json") as f:
        index = json.load(f)
    rows = [{"sequence_id": int(sid), **v} for sid, v in index.items()]
    full_meta = pd.DataFrame(rows)
    if full_meta.empty:
        sys.exit("Empty index.json")

    dataset = SignsDataset(args.processed, full_meta, max_frames=args.max_frames, augment=False)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=signs_collate,
        pin_memory=(DEVICE == "cuda"),
    )

    # Embed every clip.
    seq_ids = full_meta["sequence_id"].tolist()
    labels_all = full_meta["label"].tolist()
    embeddings = np.zeros((len(seq_ids), meta["d_model"]), dtype=np.float32)
    cursor = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="encoding"):
            if batch is None:
                continue
            x, _, pad_mask = batch
            x = x.to(DEVICE); pad_mask = pad_mask.to(DEVICE)
            emb = encode_features(model, x, pad_mask=pad_mask).cpu().numpy()
            embeddings[cursor : cursor + emb.shape[0]] = emb
            cursor += emb.shape[0]
    embeddings = embeddings[:cursor]
    seq_ids = seq_ids[:cursor]
    labels_all = labels_all[:cursor]

    # Per class, find the medoid in embedding space.
    print(f"Selecting medoid clips across {num_classes} classes...")
    out_signs: dict[str, dict] = {}
    clips_dir = args.processed / "clips"
    for cls in range(num_classes):
        mask = np.array([lbl == cls for lbl in labels_all], dtype=bool)
        if not mask.any():
            continue
        cls_emb = embeddings[mask]
        cls_ids = [sid for sid, m in zip(seq_ids, mask) if m]
        centroid = cls_emb.mean(axis=0)
        dists = np.linalg.norm(cls_emb - centroid, axis=1)
        best = int(np.argmin(dists))
        medoid_seq_id = cls_ids[best]

        # Load the real landmark sequence for this clip (not the
        # feature-stacked version used by the model; the frontend wants
        # raw 3D points to animate).
        npz = np.load(clips_dir / f"{medoid_seq_id}.npz")
        x = npz["x"].astype(np.float32)         # (T, 127, 3)
        missing = npz["missing"].astype(bool)    # (T, 127)
        sign_name = idx_to_sign[cls]
        out_signs[sign_name] = {
            "label": cls,
            "sequence_id": medoid_seq_id,
            "n_frames": int(x.shape[0]),
            "landmarks": x.reshape(-1).tolist(),
            "missing": missing.reshape(-1).astype(int).tolist(),
        }

    payload = {
        "format": "21 landmarks * (x, y, z), T frames, MediaPipe Holistic 127-landmark subset",
        "n_landmarks": int(x.shape[1]) if out_signs else 127,
        "n_coords": 3,
        "sign_to_idx": sign_to_idx,
        "signs": out_signs,
    }

    out_path = args.exports / "sign_references.json"
    with open(out_path, "w") as f:
        json.dump(payload, f)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path} ({size_mb:.1f} MB, {len(out_signs)} signs)")


if __name__ == "__main__":
    main()
