"""
Run a single inference against the ONNX model.
Useful for verifying the export and as a reference for integrating into openhand/backend.

Usage:
  # Random landmarks (smoke test)
  python scripts/infer.py

  # Provide a JSON file with a 63-element array
  python scripts/infer.py --landmarks path/to/landmarks.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="exports/asl_classifier.onnx", type=Path)
    parser.add_argument("--meta", default="exports/model_meta.json", type=Path)
    parser.add_argument("--landmarks", default=None, type=Path,
                        help="JSON file containing a flat array of 63 floats")
    parser.add_argument("--topk", default=3, type=int)
    args = parser.parse_args()

    if not args.onnx.exists():
        print(f"ONNX model not found at {args.onnx}")
        print("Run: python scripts/export_onnx.py")
        sys.exit(1)

    with open(args.meta) as f:
        meta = json.load(f)
    label_map = meta["label_map"]

    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    if args.landmarks:
        with open(args.landmarks) as f:
            lm = json.load(f)
        vec = np.array(lm, dtype=np.float32).reshape(1, 63)
    else:
        # Synthetic: random unit-normalized landmarks
        vec = np.random.randn(1, 63).astype(np.float32) * 0.1
        print("(Using random synthetic landmarks; prediction will be meaningless)")

    t0 = time.perf_counter()
    logits = sess.run(None, {"landmarks": vec})[0][0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs = softmax(logits)
    topk = np.argsort(probs)[::-1][:args.topk]

    print(f"\nTop-{args.topk} predictions  ({elapsed_ms:.2f} ms):")
    for rank, idx in enumerate(topk, 1):
        letter = label_map[str(idx)].upper()
        print(f"  {rank}. {letter}  {probs[idx]*100:.1f}%")


if __name__ == "__main__":
    main()
