# CTC fingerspelling transformer

Variable-length fingerspelling transcription. Trained with CTC loss on
the Kaggle ASL Fingerspelling competition data.

## Quick start

```powershell
# One-time: download the Kaggle competition data (~160 GB)
python fingerspelling/scripts/download_data.py

# Run the whole pipeline: preprocess -> train (with augment) -> export ONNX
python fingerspelling/scripts/run_pipeline.py

# Or a 2-epoch smoke run on 256 sequences:
python fingerspelling/scripts/run_pipeline.py --smoke
```

`run_pipeline.py` chains the per-step scripts below. Val CER is printed
per-epoch in the train log; the best-by-val-CER checkpoint is saved as
`best.pt` with metadata in `model_meta.json`.

## Step by step

```powershell
# 1. Get the data (~160 GB; you need a Kaggle account and accepted comp rules)
python fingerspelling/scripts/download_data.py

# 2. Preprocess: parquet shards -> per-sequence .npz files
python fingerspelling/scripts/preprocess_fingerspelling.py
# Touches each parquet shard once. ~1-2 hours.
# Writes fingerspelling/data/processed_fingerspelling/sequences/*.npz.

# 3. Train
python fingerspelling/scripts/train_ctc.py --epochs 30 --batch 16 --augment
# Held-out signers as the val set. Best checkpoint: fingerspelling/exports/best.pt
# --smoke does 2 epochs on 256 sequences to verify the pipeline end to end.

# 4. Export to ONNX
python fingerspelling/scripts/export_ctc_onnx.py
# fingerspelling/exports/asl_ctc.onnx, ~116 MB. BatchNorm is fused into the conv
# stem because the dynamo exporter can't currently handle eval-mode BN.
```

Deploy:

```powershell
copy fingerspelling\exports\asl_ctc.onnx     ..\openhand\backend\models\artifacts\
copy fingerspelling\exports\model_meta.json  ..\openhand\backend\models\artifacts\asl_ctc_meta.json
```

## Architecture

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

## Things that will bite you

- **The landmark feature ordering and normalization must match between
  training and the deployed backend exactly.**
  [model/landmarks.py](model/landmarks.py) here and
  [openhand/backend/services/ctc_landmarks.py](../../openhand/backend/services/ctc_landmarks.py)
  over there are intentional duplicates with a "must stay in sync"
  warning. There's a parity test (`../shared/dump_landmark_vectors.py` +
  the matching TS test in the frontend) that catches drift.
- **The `missing` mask is part of the contract.** Training data uses an
  explicit per-landmark bool, not zero-equality. Treating zeros as
  missing was the single most expensive bug to find.
- **The exported ONNX needs batch >= 2 to keep the batch axis dynamic.**
  A batch-1 dummy becomes a static constant during dynamo trace. The
  backend pads with an all-masked second item and discards that output.
