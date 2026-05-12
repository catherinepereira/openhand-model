# openhand-model

Training pipeline for the ASL sign classifier that powers OpenHand's backend.

Trains a small MLP on MediaPipe hand landmarks and exports it to ONNX for
CPU inference in `openhand/backend`.

## Current status (2026-05-12)

**MVP trained and exported.** ASL Alphabet classifier ready for backend integration.

| Metric | Value |
|--------|-------|
| Test accuracy | **98.95%** (held-out 5% split) |
| Val accuracy at best epoch | 98.5% |
| Training samples | 62,819 (after MediaPipe hand-detection filter) |
| Model size | 62K params, ~250 KB ONNX |
| CPU inference latency | **0.019 ms/frame** (target was <1ms) |
| Classes | 26 (A–Z) |
| Input | 63 floats (21 landmarks × xyz, wrist-centred, unit-scaled) |
| Artifacts | `exports/asl_classifier.onnx`, `exports/model_meta.json`, `exports/best.pt` |

### Pipeline history

| Stage | Status |
|-------|--------|
| Fingerspelling dataset downloaded (~160 GB) | Done — kept in `data/raw/` for future seq2seq work |
| Fingerspelling preprocessing (median-frame per character) | **Abandoned** — naive frame slicing gave ~30% accuracy because frames aren't aligned to characters (the dataset is designed for CTC sequence transcription, not per-letter classification) |
| ASL Alphabet image dataset downloaded (~1 GB) | Done |
| MediaPipe landmark extraction on 87K images | Done (62,819 with detected hands, 15,181 skipped) |
| Train MLP on alphabet landmarks | Done — 98.95% test accuracy |
| ONNX export | Done |
| Backend integration | Next |

**Why the dataset switch:** the fingerspelling dataset uses continuous video of full
phrases ("3 creekhouse", "https://jsi.is/hukuoka") with no frame-level
character labels — it's a sequence-to-sequence task. The ASL Alphabet dataset
already isolates one letter per image, which is the right shape for OpenHand's
real-time per-frame classifier. The fingerspelling data stays on disk for a
later CTC/temporal-model upgrade.

## How it fits into OpenHand

```
webcam frame
    → MediaPipe (openhand/backend/services/mediapipe_service.py)
    → 21 landmarks (63 floats, wrist-centred + scaled)
    → ONNX model (openhand/backend/models/asl_classifier.onnx)   ← trained here
    → predicted letter + confidence
    → WebSocket → frontend
```

The trained ONNX file is a drop-in replacement for the current rule-based
`SignClassifier` in `openhand/backend/services/classifier.py`.

## Structure

```
openhand-model/
  model/
    mlp.py        # ASLClassifier: 3-layer MLP [63→256→128→64→26]
    dataset.py    # ASLDataset + load_splits() with augmentation
  scripts/
    download_data.py        # Kaggle API download (fingerspelling) → data/raw/
    preprocess.py           # Fingerspelling Parquet → landmarks (abandoned for MVP)
    preprocess_alphabet.py  # ASL Alphabet images → MediaPipe landmarks → X.npy + y.npy   ← MVP path
    train.py                # Training loop → exports/best.pt
    export_onnx.py          # PyTorch → ONNX → exports/asl_classifier.onnx
    evaluate.py             # Per-class accuracy + confusion matrix
    infer.py                # Single-frame inference (smoke test)
  data/
    raw/                  # Fingerspelling Parquet files (gitignored, ~160 GB) — kept for later
    asl-alphabet/         # ASL Alphabet image dataset (gitignored, ~1 GB)
    processed/            # Fingerspelling preprocessed (abandoned, may delete)
    processed_alphabet/   # X.npy, y.npy, label_map.json   ← MVP training data
    hand_landmarker.task  # MediaPipe Tasks-API model (~8 MB)
  exports/
    best.pt                  # Best PyTorch checkpoint
    asl_classifier.onnx      # Export for backend
    model_meta.json          # num_classes, label_map, accuracy
    training_curves.png      # Loss + accuracy plots
  requirements.txt
```

## Setup

```powershell
cd openhand-model
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Full training pipeline (MVP — ASL Alphabet)

### 1. Download the ASL Alphabet dataset

```powershell
kaggle datasets download -d grassknoted/asl-alphabet -p data/asl-alphabet
Expand-Archive data/asl-alphabet/asl-alphabet.zip -DestinationPath data/asl-alphabet
```

87,000 labelled images, 3,000 per letter (A–Z + DEL/NOTHING/SPACE — we keep
only A–Z for the MVP).

### 2. Download the MediaPipe Hand Landmarker task model

```powershell
python scripts/_download_model.py
# Saves ~8 MB to data/hand_landmarker.task
```

MediaPipe 0.10+ requires this `.task` file for the new Tasks API
(`mp.solutions.hands` has been removed).

### 3. Extract landmarks

```powershell
python scripts/preprocess_alphabet.py
# Output: data/processed_alphabet/X.npy  (N, 63)  float32
#         data/processed_alphabet/y.npy  (N,)     int64
#         data/processed_alphabet/label_map.json
# Also writes data/processed_alphabet/_per_letter/*.npz checkpoints
# so re-running resumes from where it left off.
```

Runs MediaPipe Hands on each image, keeps the 63-float landmark vector,
wrist-centres and unit-scales it. About 30 minutes on CPU at ~50 images/sec.

### 4. Train

```powershell
python scripts/train.py --data data/processed_alphabet --epochs 60 --batch 512
# Best checkpoint saved to exports/best.pt
# Training curves saved to exports/training_curves.png
```

GPU is used automatically if available. On CPU only, reduce `--epochs` to 20
for a quick sanity check.

### 4. Export to ONNX

```powershell
python scripts/export_onnx.py
# Output: exports/asl_classifier.onnx
# Prints avg CPU inference latency (target: <1ms/frame)
```

### 5. Evaluate

```powershell
python scripts/evaluate.py
# Per-letter accuracy, confusion matrix
```

### 6. Deploy to OpenHand

```powershell
Copy-Item exports/asl_classifier.onnx ..\openhand\backend\models\
Copy-Item exports/model_meta.json     ..\openhand\backend\models\
```

Then update `openhand/backend/services/classifier.py` to load the ONNX model
via `onnxruntime` instead of the rule-based heuristics (see integration notes below).

## Model details

| Component | Detail |
|-----------|--------|
| Input | 63 floats — 21 MediaPipe hand landmarks × (x, y, z), wrist at origin, scaled to unit range |
| Architecture | Linear(63→256) → BN → ReLU → Dropout(0.3) → Linear(256→128) → BN → ReLU → Dropout(0.3) → Linear(128→64) → BN → ReLU → Dropout(0.3) → Linear(64→26) |
| Output | 26 logits (A–Z) |
| Loss | CrossEntropyLoss with label_smoothing=0.05 |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR |
| Augmentation | Gaussian noise σ=0.01, scale jitter ±5% |
| Target accuracy | >95% top-1 on held-out signers |

## Integrating the ONNX model into openhand/backend

Replace the body of `SignClassifier.classify()` in
`openhand/backend/services/classifier.py` with:

```python
import onnxruntime as ort
import numpy as np, json
from pathlib import Path

_META = json.loads((Path(__file__).parent.parent / "models/model_meta.json").read_text())
_SESS = ort.InferenceSession(
    str(Path(__file__).parent.parent / "models/asl_classifier.onnx"),
    providers=["CPUExecutionProvider"],
)
_LABEL_MAP = _META["label_map"]   # {"0": "a", "1": "b", ...}

def classify(self, landmarks) -> Optional[DetectionResult]:
    if landmarks is None or len(landmarks) != 21:
        return None
    vec = self._to_vec(landmarks)          # existing landmarks_to_array equivalent
    logits = _SESS.run(None, {"landmarks": vec.reshape(1, 63)})[0][0]
    probs = np.exp(logits) / np.exp(logits).sum()
    idx = int(probs.argmax())
    return DetectionResult(
        sign=_LABEL_MAP[str(idx)].upper(),
        confidence=float(probs[idx]),
        landmarks=landmarks,
    )
```

Add `onnxruntime` to `openhand/backend/requirements.txt`.

## Dataset notes

**MVP (current)**: [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — 87K images, 26 letter classes (3,000/letter), single hand per image, varied lighting. Per-frame classification — matches OpenHand's real-time inference shape exactly.

**Future seq2seq upgrade**: [ASL Fingerspelling](https://www.kaggle.com/competitions/asl-fingerspelling) — 120K+ sequences of full phrases with pre-extracted MediaPipe Holistic landmarks (Parquet). Already downloaded to `data/raw/`. Requires a temporal model (LSTM / 1D-CNN / Transformer) and CTC loss to use properly — naive frame-to-character slicing does not work (we tried, got 30% accuracy).

**Possible expansion**: [Google ASL Signs](https://www.kaggle.com/competitions/asl-signs) — 250 isolated words, same Parquet format as fingerspelling.

## Known limitations

- **Single-frame, no temporal context.** J and Z require motion to distinguish from I and D respectively; a static-frame model will confuse them. Mark these as "needs motion" in the UI until a temporal model is added.
- **Single dominant hand.** Left-handed signers may see lower accuracy until we mirror-augment.
- **ASL Alphabet dataset homogeneity.** All 87K images come from one signer with consistent lighting/background. MediaPipe landmark normalisation removes most of the cosmetic variance, but real-world accuracy will be lower than test-set accuracy until we add cross-signer data (the fingerspelling dataset, once we have a temporal model, has 100+ signers).
