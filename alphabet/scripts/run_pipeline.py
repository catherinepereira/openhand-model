"""
End-to-end alphabet model pipeline:
  preprocess -> train -> evaluate -> export ONNX -> build Learn references.

Assumes the dataset is already downloaded (run download_alphabet_data.py once).
Each step is also runnable on its own; this script just chains them so the
common case is a single command.

Usage:
  python alphabet/scripts/run_pipeline.py
  python alphabet/scripts/run_pipeline.py --epochs 30 --batch 1024
  python alphabet/scripts/run_pipeline.py --skip-preprocess  # data already extracted
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
    ap.add_argument("--batch", default=512, type=int)
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-references", action="store_true")
    args = ap.parse_args()

    if not args.skip_preprocess:
        run("Preprocess", [PY, str(SCRIPTS / "preprocess_alphabet.py")])

    run("Train", [
        PY, str(SCRIPTS / "train.py"),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
    ])

    if not args.skip_eval:
        run("Evaluate", [PY, str(SCRIPTS / "evaluate.py")])

    run("Export ONNX", [PY, str(SCRIPTS / "export_onnx.py")])

    if not args.skip_references:
        run("Build Learn references", [PY, str(SCRIPTS / "build_reference_landmarks.py")])

    print("\nPipeline complete. Artifacts in:", MODEL_ROOT / "exports")


if __name__ == "__main__":
    main()
