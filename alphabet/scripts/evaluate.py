"""
Evaluate a trained model (PyTorch or ONNX) on the held-out test split.
Prints per-class accuracy and a confusion matrix.

Usage:
  # PyTorch checkpoint
  python scripts/evaluate.py --checkpoint exports/best.pt

  # ONNX model
  python scripts/evaluate.py --onnx exports/asl_classifier.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_ROOT = Path(__file__).resolve().parent.parent


def run_pytorch(checkpoint: Path, meta_path: Path, X_test: np.ndarray, y_test: np.ndarray):
    import torch
    from model.mlp import build_model

    with open(meta_path) as f:
        meta = json.load(f)

    model = build_model(num_classes=meta["num_classes"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float())
        preds = logits.argmax(1).numpy()

    return preds, meta["label_map"]


def run_onnx(onnx_path: Path, meta_path: Path, X_test: np.ndarray, y_test: np.ndarray):
    import onnxruntime as ort

    with open(meta_path) as f:
        meta = json.load(f)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"landmarks": X_test.astype(np.float32)})[0]
    preds = logits.argmax(axis=1)
    return preds, meta["label_map"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=MODEL_ROOT / "data" / "processed_alphabet", type=Path)
    parser.add_argument("--meta", default=MODEL_ROOT / "exports" / "model_meta.json", type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path)
    parser.add_argument("--onnx", default=None, type=Path)
    args = parser.parse_args()

    if args.checkpoint is None and args.onnx is None:
        if (MODEL_ROOT / "exports" / "asl_classifier.onnx").exists():
            args.onnx = MODEL_ROOT / "exports" / "asl_classifier.onnx"
        else:
            args.checkpoint = MODEL_ROOT / "exports" / "best.pt"

    X = np.load(args.data / "X.npy")
    y = np.load(args.data / "y.npy")
    from sklearn.model_selection import train_test_split
    _, X_tv, _, y_tv = train_test_split(X, y, test_size=0.95, random_state=42, stratify=y)
    _, X_test, _, y_test = train_test_split(X_tv, y_tv, test_size=0.05/0.95, random_state=42, stratify=y_tv)

    if args.onnx:
        print(f"Evaluating ONNX model: {args.onnx}")
        preds, label_map = run_onnx(args.onnx, args.meta, X_test, y_test)
    else:
        print(f"Evaluating PyTorch checkpoint: {args.checkpoint}")
        preds, label_map = run_pytorch(args.checkpoint, args.meta, X_test, y_test)

    labels = [label_map[str(i)].upper() for i in range(len(label_map))]

    acc = (preds == y_test).mean()
    print(f"\nOverall accuracy: {acc:.4f}  ({acc*100:.2f}%)\n")
    print(classification_report(y_test, preds, target_names=labels, zero_division=0))

    # Confusion matrix (show only rows with errors)
    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix (letters with >0 errors):")
    for i, row in enumerate(cm):
        if row[i] < row.sum():
            errors = [(labels[j], int(row[j])) for j in range(len(labels)) if j != i and row[j] > 0]
            errors.sort(key=lambda x: -x[1])
            err_str = ", ".join(f"{l}:{n}" for l, n in errors[:5])
            print(f"  {labels[i]}: correct={row[i]}/{row.sum()}  confused_with=[{err_str}]")


if __name__ == "__main__":
    main()
