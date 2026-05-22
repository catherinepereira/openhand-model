"""
End-to-end CTC fingerspelling pipeline:
  preprocess -> train -> export ONNX.

Assumes the dataset is already downloaded (run download_fingerspelling_data.py once).
Each step is also runnable on its own; this script just chains them so the
common case is a single command.

Usage:
  python fingerspelling/scripts/run_pipeline.py
  python fingerspelling/scripts/run_pipeline.py --epochs 50 --batch 32
  python fingerspelling/scripts/run_pipeline.py --skip-preprocess
  python fingerspelling/scripts/run_pipeline.py --smoke  # quick sanity check
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = MODEL_ROOT / "scripts"
PY = sys.executable


def run(name: str, cmd: list[str]) -> None:
    print(f"\n=== {name} ===")
    print(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nStep '{name}' failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", default=30, type=int)
    ap.add_argument("--batch", default=16, type=int)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    if not args.skip_preprocess:
        run("Preprocess", [PY, str(SCRIPTS / "preprocess_fingerspelling.py")])

    train_cmd = [
        PY, str(SCRIPTS / "train_ctc.py"),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
    ]
    if not args.no_augment:
        train_cmd.append("--augment")
    if args.smoke:
        train_cmd.append("--smoke")
    run("Train", train_cmd)

    if not args.skip_export:
        run("Export ONNX", [PY, str(SCRIPTS / "export_ctc_onnx.py")])

    print("\nPipeline complete. Artifacts in:", MODEL_ROOT / "exports")
    print("Val CER is reported per-epoch in the train log; the best checkpoint")
    print("is saved as best.pt with metadata in model_meta.json.")


if __name__ == "__main__":
    main()
