"""
Export the trained CTC transformer to ONNX for CPU inference in the backend.

The model takes a variable-length landmark sequence (B, T, N_FEATURES) plus a
boolean padding mask (B, T), and returns log-probs (T, B, V). We export with
dynamic axes on both batch and time so the backend can send any-length
sequences.

Usage:
  python scripts/export_ctc_onnx.py
  python scripts/export_ctc_onnx.py --checkpoint exports/ctc/best.pt \
                                    --meta       exports/ctc/model_meta.json \
                                    --out        exports/ctc/asl_ctc.onnx
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# torch.onnx prints unicode glyphs that crash on Windows cp1252 consoles
# mid-export. Force UTF-8 IO before importing torch.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.ctc_transformer import build_ctc_model
from model.landmarks import N_FEATURES

MODEL_ROOT = Path(__file__).resolve().parent.parent


def _fuse_conv_bn_in_place(seq: torch.nn.Sequential) -> None:
    """Replace adjacent Conv1d->BatchNorm1d pairs in a Sequential with a
    single fused Conv1d (BN folded into weights/bias)."""
    new_children = []
    skip_next = False
    items = list(seq.named_children())
    for i, (_, child) in enumerate(items):
        if skip_next:
            skip_next = False
            continue
        if isinstance(child, torch.nn.Conv1d) and i + 1 < len(items) and isinstance(items[i + 1][1], torch.nn.BatchNorm1d):
            fused = torch.nn.utils.fusion.fuse_conv_bn_eval(child, items[i + 1][1])
            new_children.append(fused)
            skip_next = True
        else:
            new_children.append(child)
    seq._modules.clear()
    for idx, mod in enumerate(new_children):
        seq.add_module(str(idx), mod)


class CTCExportWrapper(torch.nn.Module):
    """Wrapper that exposes the pad mask as a positional bool tensor;
    onnx.export doesn't handle the original keyword-only signature."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        return self.inner(x, src_key_padding_mask=pad_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=MODEL_ROOT / "exports" / "best.pt", type=Path)
    ap.add_argument("--meta",       default=MODEL_ROOT / "exports" / "model_meta.json", type=Path)
    ap.add_argument("--out",        default=MODEL_ROOT / "exports" / "asl_ctc.onnx", type=Path)
    ap.add_argument("--opset",      default=17, type=int)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}. Run train_ctc.py first.")

    with open(args.meta) as f:
        meta = json.load(f)

    num_classes = meta["num_classes"]
    d_model     = meta["d_model"]
    nhead       = meta["nhead"]
    layers      = meta["layers"]

    inner = build_ctc_model(
        num_classes=num_classes, d_model=d_model, nhead=nhead, num_layers=layers,
    )
    inner.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    inner.eval()

    # Fold BatchNorm1d into the preceding Conv1d; the dynamo exporter can't
    # currently convert eval-mode BatchNorm.
    _fuse_conv_bn_in_place(inner.stem)

    model = CTCExportWrapper(inner).eval()

    # batch=2 because dynamo treats a batch-1 dummy as a static constant.
    dummy_x = torch.zeros(2, 64, N_FEATURES, dtype=torch.float32)
    dummy_mask = torch.zeros(2, 64, dtype=torch.bool)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {args.out} (opset {args.opset}, dynamo) ...")
    dynamic_shapes = {
        "x":         {0: torch.export.Dim("batch", max=64), 1: torch.export.Dim("time", max=1024)},
        "pad_mask":  {0: torch.export.Dim("batch", max=64), 1: torch.export.Dim("time", max=1024)},
    }
    onnx_program = torch.onnx.export(
        model,
        (dummy_x, dummy_mask),
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        input_names=["landmarks", "pad_mask"],
        output_names=["log_probs"],
    )
    onnx_program.save(str(args.out))

    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])

    T = 128
    x = np.random.randn(1, T, N_FEATURES).astype(np.float32)
    pad = np.zeros((1, T), dtype=bool)
    out = sess.run(None, {"landmarks": x, "pad_mask": pad})[0]
    print(f"Output shape: {out.shape}  (expected: ({T}, 1, {num_classes}))")
    assert out.shape == (T, 1, num_classes)

    runs = 50
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"landmarks": x, "pad_mask": pad})
    elapsed_ms = (time.perf_counter() - t0) / runs * 1000
    print(f"Avg inference for T={T}: {elapsed_ms:.1f} ms (CPU)")

    print(f"\nOutput: {args.out}")
    print("Next step: copy to backend artifacts and wire up the route.")


if __name__ == "__main__":
    main()
