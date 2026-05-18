"""
Preprocess the ASL Alphabet image dataset into landmark feature vectors.

For each image:
  1. Run MediaPipe Hands to extract 21 hand landmarks (63 floats)
  2. Normalize: wrist at origin, scale by 95th-percentile abs value
  3. Save as X.npy / y.npy

Dataset layout expected at --images:
  asl_alphabet_train/
    A/ *.jpg
    B/ *.jpg
    ...
    del/  (DELETE)
    nothing/
    space/

We only keep A-Z (26 classes) and skip del/nothing/space; OpenHand only
needs letter recognition for the MVP.

Output (data/processed_alphabet/):
  X.npy           float32 (N, 63)
  y.npy           int64   (N,)
  label_map.json  {"0": "A", "1": "B", ...}

Usage:
  python scripts/preprocess_alphabet.py
  python scripts/preprocess_alphabet.py --images data/asl-alphabet/asl_alphabet_train/asl_alphabet_train
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from tqdm import tqdm

LETTERS = [chr(ord("A") + i) for i in range(26)]
LABEL_TO_IDX = {l: i for i, l in enumerate(LETTERS)}

MODEL_ROOT = Path(__file__).resolve().parent.parent
TASK_MODEL = MODEL_ROOT.parent / "shared" / "hand_landmarker.task"


def make_detector() -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(TASK_MODEL)),
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(img_path: Path, detector) -> np.ndarray | None:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    vec = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32).flatten()
    return vec


def normalize(vec: np.ndarray) -> np.ndarray:
    pts = vec.reshape(21, 3) - vec[:3]
    scale = np.percentile(np.abs(pts), 95)
    if scale > 1e-6:
        pts = pts / scale
    return pts.reshape(63).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        default=MODEL_ROOT / "data" / "asl-alphabet" / "asl_alphabet_train" / "asl_alphabet_train",
        type=Path,
    )
    parser.add_argument("--out", default=MODEL_ROOT / "data" / "processed_alphabet", type=Path)
    args = parser.parse_args()

    if not args.images.exists():
        # Try one level up (zip may extract differently)
        alt = args.images.parent.parent / "asl_alphabet_train"
        if alt.exists():
            args.images = alt
        else:
            raise FileNotFoundError(
                f"Image directory not found: {args.images}\n"
                "Run: python scripts/preprocess_alphabet.py --images <path to asl_alphabet_train folder>"
            )

    args.out.mkdir(parents=True, exist_ok=True)

    all_x, all_y = [], []
    failed = 0
    ckpt_dir = args.out / "_per_letter"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with make_detector() as detector:
        for letter in LETTERS:
            ckpt_file = ckpt_dir / f"{letter}.npz"
            if ckpt_file.exists():
                d = np.load(ckpt_file)
                lx, ly = d["x"], d["y"]
                lfailed = int(d["failed"])
                all_x.extend(lx)
                all_y.extend(ly.tolist())
                failed += lfailed
                print(f"[{letter}] resumed from cache: {len(lx)} samples, {lfailed} failed")
                continue

            folder = args.images / letter
            if not folder.exists():
                print(f"[{letter}] Warning: folder not found: {folder}")
                continue

            images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            label_idx = LABEL_TO_IDX[letter]

            lx, ly = [], []
            lfailed = 0
            for img_path in tqdm(images, desc=f"[{letter}]", leave=False):
                vec = extract_landmarks(img_path, detector)
                if vec is None:
                    lfailed += 1
                    continue
                vec = normalize(vec)
                lx.append(vec)
                ly.append(label_idx)

            np.savez(
                ckpt_file,
                x=np.stack(lx).astype(np.float32) if lx else np.zeros((0, 63), np.float32),
                y=np.array(ly, dtype=np.int64),
                failed=lfailed,
            )
            print(f"[{letter}] done: {len(lx)} samples, {lfailed} failed (checkpointed)")
            all_x.extend(lx)
            all_y.extend(ly)
            failed += lfailed

    print(f"\nExtracted {len(all_x)} samples  ({failed} images had no hand detected)")

    X = np.stack(all_x).astype(np.float32)
    y = np.array(all_y, dtype=np.int64)

    print(f"Shape: {X.shape}")
    print("\nPer-letter counts:")
    for i, letter in enumerate(LETTERS):
        count = int((y == i).sum())
        print(f"  {letter}: {count}")

    np.save(args.out / "X.npy", X)
    np.save(args.out / "y.npy", y)

    label_map = {str(i): l for i, l in enumerate(LETTERS)}
    with open(args.out / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nSaved to {args.out}/")
    print("Next: python scripts/train.py --data data/processed_alphabet --epochs 60")


if __name__ == "__main__":
    main()
