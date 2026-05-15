"""
Fetch the MediaPipe Hand Landmarker .task file into data/.

Run once after cloning the repo, before preprocess_alphabet.py. Idempotent:
re-running with the file already present is a no-op.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEST = Path(__file__).resolve().parent.parent / "data" / "hand_landmarker.task"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        size_mb = DEST.stat().st_size / 1e6
        print(f"OK    {DEST.name} ({size_mb:.1f} MB) already present")
        return
    print(f"DL    {URL}")
    with urllib.request.urlopen(URL) as resp:
        DEST.write_bytes(resp.read())
    size_mb = DEST.stat().st_size / 1e6
    print(f"saved {DEST.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
