# CTC Training — Current State

> Status doc for the fingerspelling transformer + CTC training effort.
> Edit as runs complete or plans change.

## TL;DR

A Transformer + CTC sequence model trained on the Kaggle ASL Fingerspelling
dataset (67K sequences, 100+ signers) to handle phrase transcription —
something the existing per-frame alphabet MLP can't do. Architecture and
pipeline are validated end-to-end. Two real training runs done so far:

| Run | Architecture | Aug | Epochs | Best val CER |
|-----|--------------|-----|--------|-------------:|
| 1 (`bezl4y74y`) | 5.5M params (d=256, 6L) | mild | 40 | **0.274** |
| 2 (`bc1cgh3au`) | 27.5M params (d=512, 12L) | mild | 100 | **0.248** |

For comparison the Kaggle 1st place solution got ~0.21 on the held-out test
set using a similar-size model + beam search decoding. We're close on
greedy decoding alone.

## Why CTC and not the alphabet MLP

The existing alphabet model (`exports/asl_classifier.onnx`, 98.95% test acc)
was trained on still photos from a single signer. It works in the live demo
but struggles on:

- J and Z (motion-defined letters; the static-image data only captured
  arbitrary moments mid-motion)
- Letters that look similar from a fixed angle (M/N/S/T, U/V, P/Q)
- Signers / lighting / camera angles different from the training data

The fingerspelling dataset solves both: 100+ signers AND multi-frame
sequences where J/Z motions appear naturally. But it requires sequence-to-
sequence training (CTC) because the dataset has no per-frame alignments —
just (sequence, target string) pairs.

## Components (all built and validated)

| File | Purpose |
|------|---------|
| `model/landmarks.py` | 130-landmark selection + per-sequence normalisation. Mirrors the Kaggle 1st-place feature set: 40 lips + 32 eyes + 4 nose + 9 pose + 42 hands × 3 axes = 390 floats/frame |
| `model/ctc_transformer.py` | Conv1d stem → sinusoidal PE → Transformer encoder → linear CTC head. Configurable d_model / layers / heads |
| `model/fingerspelling_dataset.py` | Dataset over per-sequence `.npz` files (no Parquet at train time) + 6 augmentation tricks |
| `scripts/preprocess_fingerspelling.py` | One-time extraction Parquet → `.npz`. ~10 min for full 67K seq, produces ~11 GB |
| `scripts/train_ctc.py` | Training loop with AMP, cosine LR + warmup, grad clip, signer-held-out val, greedy decode, CER metric. `--smoke`, `--subset N`, `--augment` flags |
| `scripts/export_ctc_onnx.py` | Export PyTorch → ONNX with dynamic batch + time axes. Folds BN into conv weights for cleaner export |

## IN-FLIGHT: stronger augmentation — task `bhkk7telo`

- Architecture: same as run 2 (d=512, 12L, 27.5M params, batch=48).
- Run 2 found that capacity wasn't the binding constraint — train loss
  reached 0.62 while val loss plateaued at 1.10, suggesting overfit.
- New run keeps the architecture but adds **6 augmentation tricks** vs the
  previous 2, to break the overfit ceiling:
  1. Wider time crop range (70-100% of frames, was 85-100%)
  2. Time stretch (0.85-1.15× speed via frame resampling)
  3. Landmark group dropout (8% chance to zero entire face/pose/hand)
  4. Affine jitter (scale ±10%, translate ±0.05)
  5. Per-frame masking (5%, same as before)
  6. SpecAugment-style contiguous time-mask spans (1-2 spans, up to 10% of T)
- Epochs reduced 100 → 80 since loss curves were plateauing toward the end.
- ETA: ~3.5 hours.
- Log: `C:\Users\cathe\AppData\Local\Temp\claude\c--Users-cathe-Developer\a95a387c-4d62-42b8-ba58-d47cfcd59638\tasks\bhkk7telo.output`
- Command:
  ```
  python scripts/train_ctc.py --epochs 80 --batch 48 --workers 2 \
    --d_model 512 --nhead 8 --layers 12 --max_frames 256 \
    --lr 1e-3 --weight_decay 0.05 --warmup_steps 1500 \
    --label_smoothing 0.1 --augment
  ```

**Early signal (epoch 10)**: CER 0.421 vs run 2's ~0.49 at the same epoch.
Augmentation is the biggest lever we've pulled. If the trend continues we
should land somewhere in the 0.20-0.22 range.

## Backups & deployment

| File | What |
|------|------|
| `exports/ctc/best.pt` + `model_meta.json` | Currently-in-training (run 3) |
| `exports/ctc/best.pt.100ep_d512_l12.bak` + meta `.bak` | Run 2 weights (CER 0.248) |
| `exports/ctc/best.pt.40ep_d256_l6.bak` + meta `.bak` | Run 1 weights (CER 0.274) |
| `openhand/backend/models/artifacts/asl_ctc.onnx` | The ONNX deployed in the backend right now |

The currently-deployed ONNX is **exported from run 1 (CER 0.274)** — that's
the working CTC model the `/api/transcribe` endpoint serves.

After the in-flight run completes:
1. Run `export_ctc_onnx.py` to produce the new ONNX
2. Copy it to `openhand/backend/models/artifacts/asl_ctc.onnx`
3. Replace `asl_ctc_meta.json` too if the vocab or num_classes changed
   (they shouldn't, but check)
4. Backend hot-reloads the ONNX via uvicorn `--reload`

## After this run

1. **Inspect val examples** — compare to runs 1 & 2 to confirm the gain is
   on phrases generally, not just on the same 3 examples we print
2. **Beam search decode** — biggest remaining lever. Likely gets CER from
   wherever we land down to ~0.18-0.20. Implementation: ~50 LOC in
   `backend/services/ctc_classifier.py`, no retraining needed
3. **Mirror-augment for left-handed signers** — the dataset is biased
   toward right-handed signing. Flipping x-axis + swapping left/right hand
   landmarks doubles training samples cheaply

## Watch list / not yet done

- [ ] Beam search at inference (greedy decoding undersells the model)
- [ ] Mirror augmentation for left-handed signers
- [ ] Streaming inference: rolling-window decode during live signing
      instead of the current "hold-to-record" interaction
- [ ] Language model integration during beam search — usually halves CER
      on address/URL phrases by enforcing realistic character n-grams
- [ ] Resume from checkpoint flag (`--resume`) — currently if a run dies
      we lose the optimizer state and have to restart from epoch 1

## History (for posterity)

| Run | Task ID | Date | CER | Notes |
|-----|---------|------|-----|-------|
| Smoke (256 seq × 2 ep) | — | 2026-05-12 | 0.95 | Pipeline e2e check |
| Subset (2K seq × 8 ep) | — | 2026-05-12 | 1.00 | Blank-collapse trap (too small) |
| Subset post-audit (16K × 20 ep) | `b3okj2jpb` | 2026-05-12 | 0.516 | First non-trivial transcriptions; validated speedups |
| Full (67K × 40 ep, 5.5M) | `bezl4y74y` | 2026-05-12 | **0.274** | First "real" model. Deployed in backend |
| Full (67K × 100 ep, 27.5M) | `bc1cgh3au` | 2026-05-13 | **0.248** | Bigger model. Small win. Overfit at end |
| Full (67K × 80 ep, 27.5M, +strong aug) | `bhkk7telo` | 2026-05-13 | _in flight_ | Same arch, 6 aug tricks |

### Key bug-fixes / pipeline improvements over time

- **Per-sample Parquet re-reads** — original Dataset loaded each shard
  fresh per `__getitem__` under shuffled DataLoader; fix was the one-time
  `preprocess_fingerspelling.py` step. ~15-40× speedup.
- **Zero-as-missing sentinel** — original `normalize_sequence` treated
  zeros as missing landmarks, conflating with real near-origin coords; fix
  was an explicit boolean missing mask saved alongside the features.
- **Blank-collapse** — early CTC training collapses to "predict blank
  everywhere" as a local minimum. Fixed with KL-to-uniform label smoothing
  (weight 0.1) + linear LR warmup (1000-1500 steps). Smoke test immediately
  showed first-epoch hyps escaping the collapse.
- **Windows DataLoader silent crash at batch=32 workers=2** — early
  attempts at the full run died with no traceback; turned out to be a
  per-step VRAM spike (we were sometimes hitting >11 GB peak). Dropping
  batch + adding try/except around the inner loop made future crashes
  surface their tracebacks. With the strong .npz pipeline this hasn't
  recurred.
</content>
