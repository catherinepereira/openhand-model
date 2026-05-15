"""
Train the CTC transformer on the ASL Fingerspelling dataset.

Usage:
  python scripts/train_ctc.py [--epochs 30] [--batch 16] [--smoke]

--smoke uses only ~256 sequences and 2 epochs for a quick end-to-end check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.ctc_transformer import build_ctc_model  # noqa: E402
from model.fingerspelling_dataset import (  # noqa: E402
    FingerspellingDataset,
    ctc_collate,
    load_char_vocab,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def greedy_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor, blank: int) -> list[list[int]]:
    """CTC greedy decode: argmax per frame, merge repeats, remove blanks.

    log_probs: (T, B, V), same layout as CTCLoss.
    Returns list of decoded id sequences (one per batch element).
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().numpy()
    outs = []
    for i, length in enumerate(input_lengths.tolist()):
        seq = preds[i, :length]
        collapsed = []
        prev = -1
        for v in seq:
            if v != prev:
                collapsed.append(int(v))
                prev = int(v)
        outs.append([v for v in collapsed if v != blank])
    return outs


def char_error_rate(hyps: list[str], refs: list[str]) -> float:
    """Simple Levenshtein-based CER averaged over the batch."""
    def lev(a: str, b: str) -> int:
        m, n = len(a), len(b)
        if m == 0: return n
        if n == 0: return m
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i-1] == b[j-1] else 1
                curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
            prev = curr
        return prev[n]

    total = 0
    chars = 0
    for h, r in zip(hyps, refs):
        total += lev(h, r)
        chars += max(len(r), 1)
    return total / chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",     default="data/raw",  type=Path,
                    help="Original Kaggle dump (for train.csv and vocab)")
    ap.add_argument("--processed", default="data/processed_fingerspelling", type=Path,
                    help="Pre-extracted .npz dir from preprocess_fingerspelling.py")
    ap.add_argument("--exports", default="exports/ctc", type=Path)
    ap.add_argument("--epochs",  default=30, type=int)
    ap.add_argument("--batch",   default=16, type=int)
    ap.add_argument("--lr",      default=1e-3, type=float)
    ap.add_argument("--weight_decay", default=0.05, type=float)
    ap.add_argument("--workers", default=2, type=int)
    ap.add_argument("--max_frames", default=384, type=int)
    ap.add_argument("--d_model", default=256, type=int)
    ap.add_argument("--nhead",   default=8, type=int)
    ap.add_argument("--layers",  default=6, type=int)
    ap.add_argument("--augment", action="store_true",
                    help="Enable train-time time-crop + frame-mask augmentation")
    ap.add_argument("--smoke",   action="store_true",
                    help="Tiny subset + 2 epochs to check the pipeline end-to-end")
    ap.add_argument("--subset", type=int, default=None,
                    help="Train on a random subsample of N sequences (debug runs)")
    ap.add_argument("--warmup_steps", type=int, default=500,
                    help="Linear LR warmup over the first N optimizer steps")
    ap.add_argument("--label_smoothing", type=float, default=0.1,
                    help="KL-to-uniform regularizer weight (anti-blank-collapse)")
    args = ap.parse_args()

    args.exports.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    char_to_idx, idx_to_char, num_classes = load_char_vocab(args.raw)
    blank_idx = num_classes - 1
    print(f"Vocab: {num_classes - 1} chars + 1 blank")

    seq_dir = args.processed / "sequences"
    if not seq_dir.exists() or not any(seq_dir.glob("*.npz")):
        sys.exit(
            f"Pre-extracted sequences not found at {seq_dir}.\n"
            "Run: python scripts/preprocess_fingerspelling.py"
        )

    meta = pd.read_csv(args.raw / "train.csv")
    meta = meta[meta["phrase"].str.len() > 0]

    if args.smoke:
        meta = meta.sample(n=256, random_state=42).reset_index(drop=True)
        args.epochs = 2
    elif args.subset is not None:
        meta = meta.sample(n=min(args.subset, len(meta)), random_state=42).reset_index(drop=True)

    available = {int(p.stem) for p in seq_dir.glob("*.npz")}
    before = len(meta)
    meta = meta[meta["sequence_id"].isin(available)].reset_index(drop=True)
    if len(meta) < before:
        print(f"  Filtered {before - len(meta)} sequences without .npz files")

    # Split by participant so val signers are held out.
    rng = np.random.default_rng(42)
    participants = meta["participant_id"].unique()
    rng.shuffle(participants)
    n_val = max(1, int(0.05 * len(participants)))
    val_ids   = set(participants[:n_val].tolist())
    train_meta = meta[~meta["participant_id"].isin(val_ids)].reset_index(drop=True)
    val_meta   = meta[ meta["participant_id"].isin(val_ids)].reset_index(drop=True)
    print(f"Train sequences: {len(train_meta)}, val sequences: {len(val_meta)} "
          f"({len(val_ids)} signers)")

    train_ds = FingerspellingDataset(
        args.processed, train_meta, max_frames=args.max_frames, augment=args.augment,
    )
    val_ds = FingerspellingDataset(
        args.processed, val_meta, max_frames=args.max_frames, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=ctc_collate,
        pin_memory=(DEVICE == "cuda"), persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=ctc_collate,
        pin_memory=(DEVICE == "cuda"), persistent_workers=(args.workers > 0),
    )

    model = build_ctc_model(
        num_classes=num_classes,
        d_model=args.d_model, nhead=args.nhead, num_layers=args.layers,
    ).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=(DEVICE == "cuda"))

    # LR schedule: linear warmup, then cosine decay, applied per optimizer step.
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    best_val_cer = float("inf")
    history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss = 0.0
        tr_n = 0
        batch_idx = 0
        for batch in train_loader:
            batch_idx += 1
            if batch_idx % 50 == 0:
                print(f"  e{epoch} step {batch_idx}/{steps_per_epoch}  "
                      f"loss={tr_loss/max(tr_n,1):.4f}  lr={lr_at(global_step):.2e}",
                      flush=True)
            if batch is None:
                continue
            try:
                x, y, in_lens, tgt_lens, pad_mask = batch
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                in_lens = in_lens.to(DEVICE)
                tgt_lens = tgt_lens.to(DEVICE)
                pad_mask = pad_mask.to(DEVICE)

                for pg in optimizer.param_groups:
                    pg["lr"] = lr_at(global_step)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                    log_probs = model(x, src_key_padding_mask=pad_mask)
                    ctc_loss = criterion(log_probs, y, in_lens, tgt_lens)
                    # KL-to-uniform on non-pad positions to discourage blank collapse.
                    if args.label_smoothing > 0:
                        valid = (~pad_mask).transpose(0, 1).unsqueeze(-1)
                        uniform_kl = -(log_probs.mean(dim=-1))
                        smoothing = (uniform_kl.unsqueeze(-1) * valid).sum() / valid.sum().clamp_min(1)
                        loss = ctc_loss + args.label_smoothing * smoothing
                    else:
                        loss = ctc_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                tr_loss += ctc_loss.item() * x.size(0)
                tr_n += x.size(0)
                global_step += 1
            except Exception:
                print(f"\n!!! CRASH at epoch {epoch}, step {batch_idx} !!!", flush=True)
                print(f"  Batch shape: x={tuple(x.shape) if 'x' in dir() else '?'}, "
                      f"in_lens={in_lens.tolist() if 'in_lens' in dir() else '?'}", flush=True)
                traceback.print_exc()
                raise

        tr_loss /= max(tr_n, 1)

        # Validation
        model.eval()
        vl_loss, vl_n = 0.0, 0
        all_hyps, all_refs = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                x, y, in_lens, tgt_lens, pad_mask = batch
                x = x.to(DEVICE); y = y.to(DEVICE)
                in_lens = in_lens.to(DEVICE); tgt_lens = tgt_lens.to(DEVICE)
                pad_mask = pad_mask.to(DEVICE)
                log_probs = model(x, src_key_padding_mask=pad_mask)
                loss = criterion(log_probs, y, in_lens, tgt_lens)
                vl_loss += loss.item() * x.size(0)
                vl_n += x.size(0)

                decoded = greedy_decode(log_probs, in_lens, blank=blank_idx)
                refs = []
                offset = 0
                for L in tgt_lens.tolist():
                    refs.append(y[offset:offset+L].tolist())
                    offset += L
                for hyp_ids, ref_ids in zip(decoded, refs):
                    all_hyps.append("".join(idx_to_char.get(i, "?") for i in hyp_ids))
                    all_refs.append("".join(idx_to_char.get(i, "?") for i in ref_ids))

        vl_loss /= max(vl_n, 1)
        cer = char_error_rate(all_hyps, all_refs) if all_hyps else float("nan")
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": vl_loss, "val_cer": cer})

        improved = cer < best_val_cer
        if improved:
            best_val_cer = cer
            torch.save(model.state_dict(), args.exports / "best.pt")

        dt = time.time() - t0
        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {tr_loss:.4f} | val {vl_loss:.4f} | CER {cer:.3f}{marker} | "
            f"{dt:.0f}s"
        )

        if all_hyps:
            print("  examples:")
            for h, r in list(zip(all_hyps, all_refs))[:3]:
                print(f"    ref={r!r}  hyp={h!r}")

    with open(args.exports / "model_meta.json", "w") as f:
        json.dump({
            "num_classes": num_classes,
            "blank_idx": blank_idx,
            "char_to_idx": char_to_idx,
            "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
            "d_model": args.d_model,
            "nhead": args.nhead,
            "layers": args.layers,
            "max_frames": args.max_frames,
            "best_val_cer": best_val_cer,
            "history": history,
        }, f, indent=2)
    print(f"\nBest val CER: {best_val_cer:.3f}")
    print(f"Saved to {args.exports}/")


if __name__ == "__main__":
    main()
