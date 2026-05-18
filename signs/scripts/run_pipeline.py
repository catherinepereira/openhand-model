"""
End-to-end isolated-sign classifier pipeline:
  preprocess -> train -> export ONNX -> build sign references.

Assumes the dataset is already downloaded (run download_signs_data.py once).
Each step is also runnable on its own; this script just chains them so the
common case is a single command.

Usage:
  python signs/scripts/run_pipeline.py
  python signs/scripts/run_pipeline.py --epochs 80 --batch 128
  python signs/scripts/run_pipeline.py --skip-preprocess
  python signs/scripts/run_pipeline.py --smoke  # quick sanity check
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
    ap.add_argument("--epochs", default=60, type=int)
    ap.add_argument("--batch", default=64, type=int)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-references", action="store_true")
    args = ap.parse_args()

    if not args.skip_preprocess:
        run("Preprocess", [PY, str(SCRIPTS / "preprocess_signs.py")])

    train_cmd = [
        PY, str(SCRIPTS / "train_signs.py"),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
    ]
    if not args.no_augment:
        train_cmd.append("--augment")
    if args.smoke:
        train_cmd.append("--smoke")
    run("Train", train_cmd)

    if not args.skip_export:
        run("Export ONNX", [PY, str(SCRIPTS / "export_signs_onnx.py")])

    if not args.skip_references:
        run("Build sign references", [PY, str(SCRIPTS / "build_sign_references.py")])

    print("\nPipeline complete. Artifacts in:", MODEL_ROOT / "exports")
    print("Val top-1 and top-5 are reported per-epoch in the train log;")
    print("best.pt + model_meta.json hold the best-by-val-acc checkpoint.")


if __name__ == "__main__":
    main()
