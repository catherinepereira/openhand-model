"""
Download the Kaggle ASL Alphabet dataset.

Prerequisites:
  1. pip install kaggle
  2. Place kaggle.json at ~/.kaggle/kaggle.json (kaggle.com -> Account -> API)

Downloads to alphabet/data/asl-alphabet/ (~1 GB unzipped).
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = MODEL_ROOT / "data" / "asl-alphabet"
DATASET = "grassknoted/asl-alphabet"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    inner = RAW_DIR / "asl_alphabet_train"
    if inner.exists():
        print(f"OK    {inner} already exists; skipping download")
        return

    print(f"Downloading {DATASET} to {RAW_DIR} ...")
    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", DATASET,
        "-p", str(RAW_DIR),
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("\nDownload failed. Common causes:")
        print("  - kaggle.json not found at ~/.kaggle/kaggle.json")
        print("  - You haven't accepted the dataset terms at:")
        print(f"    https://www.kaggle.com/datasets/{DATASET}")
        sys.exit(1)

    print("Unzipping ...")
    for zf in RAW_DIR.glob("*.zip"):
        print(f"  {zf.name}")
        with zipfile.ZipFile(zf) as z:
            z.extractall(RAW_DIR)
        zf.unlink()

    print(f"\nDone. Raw data at: {RAW_DIR}")
    print("Next: python alphabet/scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
