# Isolated-sign classifier

250-class classification on the [Google ISLR](https://www.kaggle.com/competitions/asl-signs) competition data. 
Architecture follows the 1st-place solution's family:
1D-Conv stem over the feature axis + Transformer encoder with
attention over frames + masked global-average-pool + classification
head. 
Around 3-5M params depending on `--d_model` / `--layers`.

## Quick start

```powershell
# One-time: download the Kaggle competition data (~5 GB)
python signs/scripts/download_signs_data.py

# Run the whole pipeline: preprocess -> train (with augment) -> export -> references
python signs/scripts/run_pipeline.py

# Or a 2-epoch smoke run on 512 clips:
python signs/scripts/run_pipeline.py --smoke
```

`run_pipeline.py` chains the per-step scripts below. Val top-1 and top-5
are printed per-epoch in the train log; `best.pt` + `model_meta.json`
hold the best-by-val-acc checkpoint.

## Step by step

```powershell
# 1. Get the data (~5 GB; you need a Kaggle account and accepted comp rules)
python signs/scripts/download_signs_data.py

# 2. Preprocess: parquet shards -> per-clip .npz files
python signs/scripts/preprocess_signs.py
# Writes signs/data/processed_signs/clips/*.npz plus sign_to_idx.json + index.json.

# 3. Train
python signs/scripts/train_signs.py --epochs 60 --batch 64 --augment
# Held-out signers as val. Best checkpoint: signs/exports/best.pt.
# --smoke does 2 epochs on 512 clips for a pipeline sanity check.

# 4. Export to ONNX
python signs/scripts/export_signs_onnx.py
# signs/exports/sign_classifier.onnx, ~10-20 MB. Conv-BN fusion applied
# because the dynamo exporter can't handle eval-mode BN cleanly.

# 5. Build the Learn-the-words references (medoid clip per class)
python signs/scripts/build_sign_references.py
# signs/exports/sign_references.json: for each of 250 signs, the real
# training clip closest to that class's centroid in the trained
# encoder's embedding space. The frontend animates the medoid clip as
# the reference for that sign.
```

Deploy:

```powershell
copy signs\exports\sign_classifier.onnx  ..\openhand\backend\models\artifacts\
copy signs\exports\model_meta.json       ..\openhand\backend\models\artifacts\sign_classifier_meta.json
copy signs\exports\sign_references.json  ..\openhand\backend\models\artifacts\
```

## Architecture

| | |
|-|-|
| Input | (T, N_FEATURES=804) per clip: 381 base (127 landmarks * x/y/z) + 381 motion deltas + 42 hand-to-lip distances |
| Stem | Two 1D-conv blocks (kernel 5, BN, GELU) over the feature axis |
| Encoder | 4-layer Transformer (d_model=192, nhead=8, FFN=512, pre-LN, GELU, dropout=0.2) |
| Pool | Masked global-average over time |
| Head | Linear -> 250 logits |
| Loss | CrossEntropy with label_smoothing=0.1 |
| Optimizer | AdamW, lr=1e-3, weight_decay=0.05, linear warmup then cosine |
| Augmentation | Time crop + stretch + frame masking, mirror + hand-swap, affine (rotation up to ±30°, scale, translate), face/pose group dropout |
| Params | ~3-5M (depends on hyperparams) |

Held-out *signers* as val (not held-out clips); cross-signer
generalization is what the model needs to do at deploy time.

## Choosing references via medoid clips

Mean landmark trajectories across multiple clips of a sign make a bad reference. 
Clips vary in length, signer, framing, and MediaPipe noise, so averaging produces an unphysical blur. `build_sign_references.py` embeds every training clip with the trained encoder, computes the per-class centroid in embedding space, and picks the real clip closest to that centroid (the medoid). The frontend animates that medoid clip as the reference for the sign.
