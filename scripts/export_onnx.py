"""
Export the trained PyTorch checkpoint to ONNX for CPU inference in the backend.

Usage:
  python scripts/export_onnx.py [--checkpoint exports/best.pt] [--out exports/asl_classifier.onnx]

The resulting .onnx file is what gets copied into openhand/backend/models/
and loaded by onnxruntime in the production classifier.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.mlp import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="exports/best.pt", type=Path)
    parser.add_argument("--meta", default="exports/model_meta.json", type=Path)
    parser.add_argument("--out", default="exports/asl_classifier.onnx", type=Path)
    parser.add_argument("--opset", default=17, type=int)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run scripts/train.py first.")
        sys.exit(1)

    with open(args.meta) as f:
        meta = json.load(f)

    num_classes = meta["num_classes"]
    input_dim = meta.get("input_dim", 63)

    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.zeros(1, input_dim)

    print(f"Exporting to {args.out} (opset {args.opset}) ...")
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        opset_version=args.opset,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch_size"}, "logits": {0: "batch_size"}},
        do_constant_folding=True,
    )

    # Verify with onnxruntime
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"landmarks": dummy.numpy()})
    assert out[0].shape == (1, num_classes), f"Unexpected output shape: {out[0].shape}"

    # Smoke-test latency
    import time
    runs = 500
    x = np.random.rand(1, input_dim).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"landmarks": x})
    elapsed_ms = (time.perf_counter() - t0) / runs * 1000

    print(f"ONNX export verified. Avg inference: {elapsed_ms:.3f} ms/frame (CPU)")
    print(f"\nOutput: {args.out}")
    print(f"Label map: {meta['label_map']}")
    print(f"\nCopy to openhand backend:")
    print(f"  cp {args.out} ../openhand/backend/models/asl_classifier.onnx")
    print(f"  cp {args.meta} ../openhand/backend/models/model_meta.json")


if __name__ == "__main__":
    main()
