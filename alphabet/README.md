# Alphabet MLP

Per-frame A-Z classification on the Kaggle ASL Alphabet dataset. This is
what powers OpenHand's per-frame letter detection. 26 classes (A-Z),
single hand, ~62K params.

## Quick start

```powershell
# One-time: download the Kaggle dataset (~1 GB)
python alphabet/scripts/download_alphabet_data.py

# Run the whole pipeline: preprocess -> train -> evaluate -> export -> references
python alphabet/scripts/run_pipeline.py
```

`run_pipeline.py` chains the per-step scripts below. Pass `--epochs`,
`--batch`, or any `--skip-*` flag to customize.

## Step by step

```powershell
# 1. Get the data (~1 GB). Kaggle CLI + accepted dataset terms required.
python alphabet/scripts/download_alphabet_data.py

# 2. Run MediaPipe on every image and save 63-float landmark vectors
python alphabet/scripts/preprocess_alphabet.py
# ~30 min on CPU at ~50 images/sec. Writes alphabet/data/processed_alphabet/X.npy
# and y.npy. Per-letter checkpoints in _per_letter/ make this resumable.

# 3. Train
python alphabet/scripts/train.py --epochs 60 --batch 512
# Best checkpoint:    alphabet/exports/best.pt
# Training curves:    alphabet/exports/training_curves.png

# 4. Evaluate
python alphabet/scripts/evaluate.py
# Per-letter accuracy + confusion matrix on the held-out 5%

# 5. Export to ONNX
python alphabet/scripts/export_onnx.py
# alphabet/exports/asl_classifier.onnx plus a CPU latency benchmark

# 6. Build the Learn-screen reference from the same data
python alphabet/scripts/build_reference_landmarks.py
# alphabet/exports/reference_landmarks.json: per-letter medoid landmark vector
# (the real training sample closest to the class centroid) used by
# the Learn screen's 3D reference preview.
```

To deploy back to OpenHand:

```powershell
copy alphabet\exports\asl_classifier.onnx       ..\openhand\backend\models\artifacts\
copy alphabet\exports\model_meta.json           ..\openhand\backend\models\artifacts\
copy alphabet\exports\reference_landmarks.json  ..\openhand\backend\models\artifacts\
```

## Architecture

| | |
|-|-|
| Input | 63 floats (21 MediaPipe hand landmarks, x/y/z, wrist at origin, p95-scaled) |
| Hidden | 256 -> 128 -> 64, each with BatchNorm + ReLU + Dropout(0.3) |
| Output | 26 logits |
| Loss | CrossEntropy with `label_smoothing=0.05` |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4, cosine schedule |
| Augmentation | 3D rotation (±40° in-plane, ±10° out-of-plane), Gaussian noise σ=0.01, scale jitter ±5% |
| Params | 62,267 |
| CPU latency | 0.019 ms per frame (onnxruntime) |

The ASL Alphabet dataset is one signer in one room with consistent
lighting, so held-out test accuracy is inflated; real-world accuracy on
unseen hands is lower. Rotation augmentation in [model/dataset.py](model/dataset.py)
covers wrist tilt and palm angle so the deployed model handles P, G, H
at less-than-ideal angles.
