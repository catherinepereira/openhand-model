"""
Export the trained isolated-sign classifier to ONNX.

The model takes (B, T, N_FEATURES) features + (B, T) pad mask and returns
(B, num_classes) logits. We export with dynamic axes on batch and time
so the backend can send any-length clips up to max_len.

Usage:
  python scripts/export_signs_onnx.py [--checkpoint exports/signs/best.pt]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.sign_classifier import build_sign_classifier  # noqa: E402
from model.signs_landmarks import N_FEATURES  # noqa: E402


def _fuse_conv_bn(seq: torch.nn.Sequential) -> None:
    """Fold each Conv1d -> BatchNorm1d pair into a single Conv1d.
    Dynamo can't currently export eval-mode BN cleanly."""
    items = list(seq.named_children())
    new_children: list[torch.nn.Module] = []
    skip_next = False
    for i, (_, child) in enumerate(items):
        if skip_next:
            skip_next = False
            continue
        if (
            isinstance(child, torch.nn.Conv1d)
            and i + 1 < len(items)
            and isinstance(items[i + 1][1], torch.nn.BatchNorm1d)
        ):
            fused = torch.nn.utils.fusion.fuse_conv_bn_eval(child, items[i + 1][1])
            new_children.append(fused)
            skip_next = True
        else:
            new_children.append(child)
    seq._modules.clear()
    for idx, mod in enumerate(new_children):
        seq.add_module(str(idx), mod)


class ExportWrapper(torch.nn.Module):
    """Positional-only signature for the exporter (no kw-only args)."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        return self.inner(x, pad_mask=pad_mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="exports/signs/best.pt", type=Path)
    ap.add_argument("--meta", default="exports/signs/model_meta.json", type=Path)
    ap.add_argument("--out", default="exports/signs/sign_classifier.onnx", type=Path)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}. Run train_signs.py first.")
    with open(args.meta) as f:
        meta = json.load(f)

    num_classes = meta["num_classes"]
    d_model = meta["d_model"]
    nhead = meta["nhead"]
    layers = meta["layers"]

    inner = build_sign_classifier(
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=layers,
    )
    inner.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    inner.eval()

    _fuse_conv_bn(inner.stem)

    model = ExportWrapper(inner).eval()

    # Dynamo wants batch >= 2 for the batch axis to stay dynamic.
    dummy_x = torch.zeros(2, 32, N_FEATURES, dtype=torch.float32)
    dummy_mask = torch.zeros(2, 32, dtype=torch.bool)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {args.out} (dynamo)")
    dynamic_shapes = {
        "x": {0: torch.export.Dim("batch", max=64), 1: torch.export.Dim("time", max=256)},
        "pad_mask": {0: torch.export.Dim("batch", max=64), 1: torch.export.Dim("time", max=256)},
    }
    onnx_program = torch.onnx.export(
        model,
        (dummy_x, dummy_mask),
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        input_names=["features", "pad_mask"],
        output_names=["logits"],
    )
    onnx_program.save(str(args.out))

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    x = np.zeros((1, 64, N_FEATURES), dtype=np.float32)
    pad = np.zeros((1, 64), dtype=bool)
    out = sess.run(None, {"features": x, "pad_mask": pad})[0]
    print(f"Output shape: {out.shape}  (expected (1, {num_classes}))")
    assert out.shape == (1, num_classes)

    runs = 50
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"features": x, "pad_mask": pad})
    avg_ms = (time.perf_counter() - t0) / runs * 1000
    print(f"Avg inference for T=64: {avg_ms:.1f} ms (CPU)")
    print(f"\nOutput: {args.out}")


if __name__ == "__main__":
    main()
