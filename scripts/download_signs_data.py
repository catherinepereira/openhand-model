"""
Download the Google Isolated Sign Language Recognition competition data
from Kaggle.

You need a Kaggle account and to have accepted the competition rules at
https://www.kaggle.com/competitions/asl-signs/rules first.

Total size is ~5 GB compressed parquet, ~94,000 clips of 250 isolated
signs from ~250 signers, MediaPipe Holistic landmarks already extracted.

Usage:
  python scripts/download_signs_data.py
"""

import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_signs"
COMPETITION = "asl-signs"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {COMPETITION} to {RAW_DIR} (~5 GB)")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("kaggle package not installed. pip install kaggle.")
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        print(f"Authentication failed: {exc}")
        print("  - kaggle.json missing from ~/.kaggle/ (or %USERPROFILE%\\.kaggle\\)")
        print(f"  - or you haven't accepted the rules at:")
        print(f"    https://www.kaggle.com/competitions/{COMPETITION}/rules")
        sys.exit(1)

    try:
        api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)
    except Exception as exc:
        print(f"Download failed: {exc}")
        sys.exit(1)

    print("Unzipping...")
    for zf in RAW_DIR.glob("*.zip"):
        print(f"  {zf.name}")
        with zipfile.ZipFile(zf) as z:
            z.extractall(RAW_DIR)
        zf.unlink()

    print(f"\nDone. Raw data at: {RAW_DIR}")
    print("Next step: python scripts/preprocess_signs.py")


if __name__ == "__main__":
    main()
