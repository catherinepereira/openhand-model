# openhand-model

Training code for the two models that power [openhand](../openhand): a
small MLP for per-frame ASL alphabet classification, and a CTC
transformer for fingerspelling phrase transcription.

If you just want to run OpenHand, you don't need this repo at all once
the artifacts are built. This is the training side of the project.

## Setup

```powershell
git clone https://github.com/catherinepereira/openhand-model
cd openhand-model

python -m venv venv
venv\Scripts\Activate.ps1     # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# Fetch the MediaPipe hand landmarker (~8 MB)
python scripts/download_mediapipe_model.py
```

CUDA is detected automatically. CPU works fine for the alphabet model
(~10 min for 60 epochs). The CTC model wants a GPU; a 4090 trains an
epoch in ~17 seconds on the full dataset.

## The alphabet model

This is what powers OpenHand's per-frame letter detection. 26 classes
(A-Z), single hand, ~62K params.

```powershell
# 1. Get the data (~1 GB)
kaggle datasets download -d grassknoted/asl-alphabet -p data/asl-alphabet
Expand-Archive data/asl-alphabet/asl-alphabet.zip -DestinationPath data/asl-alphabet

# 2. Run MediaPipe on every image and save 63-float landmark vectors
python scripts/preprocess_alphabet.py
# ~30 min on CPU at ~50 images/sec. Writes data/processed_alphabet/X.npy
# and y.npy. Per-letter checkpoints in _per_letter/ make this resumable.

# 3. Train
python scripts/train.py --data data/processed_alphabet --epochs 60 --batch 512
# Best checkpoint:    exports/best.pt
# Training curves:    exports/training_curves.png

# 4. Export to ONNX
python scripts/export_onnx.py
# exports/asl_classifier.onnx plus a CPU latency benchmark

# 5. Sanity check
python scripts/evaluate.py
# Per-letter accuracy + confusion matrix on the held-out 5%

# 6. Build the Learn-screen reference from the same data
python scripts/build_reference_landmarks.py
# exports/reference_landmarks.json: per-letter mean landmark vector
# used by the Learn screen's 3D reference preview.
```

To deploy back to OpenHand:

```powershell
copy exports\asl_classifier.onnx       ..\openhand\backend\models\artifacts\
copy exports\model_meta.json           ..\openhand\backend\models\artifacts\
copy exports\reference_landmarks.json  ..\openhand\backend\models\artifacts\
```

### Architecture

| | |
|-|-|
| Input | 63 floats (21 MediaPipe hand landmarks, x/y/z, wrist at origin, p95-scaled) |
| Hidden | 256 -> 128 -> 64, each with BatchNorm + ReLU + Dropout(0.3) |
| Output | 26 logits |
| Loss | CrossEntropy with `label_smoothing=0.05` |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4, cosine schedule |
| Augmentation | Gaussian noise sigma=0.01, scale jitter +/-5% |
| Params | 62,267 |
| CPU latency | 0.019 ms per frame (onnxruntime) |

98.95% test accuracy on the held-out 5%. That number is inflated because
the ASL Alphabet dataset is one signer in one room with consistent
lighting; real-world accuracy on unseen hands is lower.

The MLP is also rotation-sensitive (it sees raw camera-frame landmarks),
which is why the deployed model occasionally needs an exaggerated angle
for letters like P, G, H. Rotation augmentation in the training loop
would help; it's a one-function change in [dataset.py](model/dataset.py)
if you want to try it.

## The CTC model

Variable-length fingerspelling transcription. Trained with CTC loss on
the Kaggle ASL Fingerspelling competition data.

```powershell
# 1. Get the data (~160 GB; you need a Kaggle account and accepted comp rules)
python scripts/download_data.py

# 2. Preprocess: parquet shards -> per-sequence .npz files
python scripts/preprocess_fingerspelling.py
# Touches each parquet shard once. ~1-2 hours.
# Writes data/processed_fingerspelling/sequences/*.npz.

# 3. Train
python scripts/train_ctc.py --epochs 30 --batch 16 --augment
# Held-out signers as the val set. Best checkpoint: exports/ctc/best.pt
# --smoke does 2 epochs on 256 sequences to verify the pipeline end to end.

# 4. Export to ONNX
python scripts/export_ctc_onnx.py
# exports/ctc/asl_ctc.onnx, ~116 MB. BatchNorm is fused into the conv
# stem because the dynamo exporter can't currently handle eval-mode BN.
```

Deploy:

```powershell
copy exports\ctc\asl_ctc.onnx        ..\openhand\backend\models\artifacts\
copy exports\ctc\model_meta.json     ..\openhand\backend\models\artifacts\asl_ctc_meta.json
```

### Architecture

| | |
|-|-|
| Input | (T, 381) per sequence: 127 landmarks (40 lips + 16 left eye + 16 right eye + 4 nose + 9 pose + 2*21 hands), 3 axes each |
| Stem | Two 1D-conv blocks (kernel 5, BN, GELU) over the feature axis |
| Encoder | 6-layer Transformer (d_model=256, nhead=8, FFN=1024, pre-LN, GELU) |
| Head | Linear -> 60 logits per frame (59 chars + blank), log_softmax |
| Loss | CTC + KL-to-uniform regularizer to discourage blank collapse |
| Optimizer | AdamW, lr=1e-3, weight_decay=0.05, linear warmup then cosine |
| Augmentation | Time crop, time stretch, frame masking, group dropout, affine jitter |
| Params | ~5.5M |
| Val CER | 0.235 (beam search width=10), 0.250 (greedy) |

### Things that will bite you

- **The landmark feature ordering and normalization must match between
  training and the deployed backend exactly.**
  [model/landmarks.py](model/landmarks.py) here and
  [openhand/backend/services/ctc_landmarks.py](../openhand/backend/services/ctc_landmarks.py)
  over there are intentional duplicates with a "must stay in sync"
  warning. There's a parity test (`scripts/dump_landmark_vectors.py` +
  the matching TS test in the frontend) that catches drift.
- **The `missing` mask is part of the contract.** Training data uses an
  explicit per-landmark bool, not zero-equality. Treating zeros as
  missing was the single most expensive bug to find.
- **The exported ONNX needs batch >= 2 to keep the batch axis dynamic.**
  A batch-1 dummy becomes a static constant during dynamo trace. The
  backend pads with an all-masked second item and discards that output.

## Tests / parity

```powershell
# Regenerate the Python<->TS fixture used by the frontend test
python scripts/dump_landmark_vectors.py
# Writes ../openhand/frontend/src/lib/__tests__/landmark_fixtures.json
```

That fixture is what catches drift between this repo's normalization
formula and the TypeScript port in the frontend.

## License

MIT. See [LICENSE](LICENSE).
