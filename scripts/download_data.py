"""
Download the ASL Fingerspelling dataset from Kaggle.

Prerequisites:
  1. pip install kaggle
  2. Place kaggle.json at ~/.kaggle/kaggle.json  (from kaggle.com -> Account -> API)
  3. Run: python scripts/download_data.py

Downloads to data/raw/ (~170 GB extracted). If you only want a quick smoke-test,
pass --sample to download just the small supplemental split (~2 GB).
"""

import argparse
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
COMPETITION = "asl-fingerspelling"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download supplemental (smaller) split only for quick testing",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {COMPETITION} to {RAW_DIR} ...")
    cmd = [
        sys.executable, "-m", "kaggle",
        "competitions", "download",
        "-c", COMPETITION,
        "-p", str(RAW_DIR),
    ]

    if args.sample:
        # Download only the supplemental_metadata.csv + a single parquet shard
        cmd += ["-f", "supplemental_metadata.csv"]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("\nDownload failed. Common causes:")
        print("  - kaggle.json not found at ~/.kaggle/kaggle.json")
        print("  - You haven't accepted the competition rules at:")
        print(f"    https://www.kaggle.com/competitions/{COMPETITION}/rules")
        sys.exit(1)

    print("Unzipping ...")
    import zipfile
    for zf in RAW_DIR.glob("*.zip"):
        print(f"  {zf.name}")
        with zipfile.ZipFile(zf) as z:
            z.extractall(RAW_DIR)
        zf.unlink()

    print(f"\nDone. Raw data at: {RAW_DIR}")
    print("Next step: python scripts/preprocess.py")


if __name__ == "__main__":
    main()
