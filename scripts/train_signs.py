"""
Train the isolated-sign classifier on the Google ISLR data.

Held-out signers (not held-out clips) as the val set. With ~250 signers
in the dataset, holding out ~5% gives a meaningful generalization
signal.

Usage:
  python scripts/train_signs.py [--epochs 60] [--batch 64] [--augment] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.sign_classifier import build_sign_classifier  # noqa: E402
from model.signs_dataset import (  # noqa: E402
    SignsDataset,
    load_sign_map,
    signs_collate,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed_signs", type=Path)
    ap.add_argument("--exports", default="exports/signs", type=Path)
    ap.add_argument("--epochs", default=60, type=int)
    ap.add_argument("--batch", default=64, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--weight_decay", default=0.05, type=float)
    ap.add_argument("--workers", default=2, type=int)
    ap.add_argument("--max_frames", default=80, type=int)
    ap.add_argument("--d_model", default=192, type=int)
    ap.add_argument("--nhead", default=8, type=int)
    ap.add_argument("--layers", default=4, type=int)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--label_smoothing", default=0.1, type=float)
    ap.add_argument("--warmup_steps", default=500, type=int)
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 512 clips for a pipeline sanity check")
    args = ap.parse_args()

    args.exports.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    sign_to_idx, idx_to_sign = load_sign_map(args.processed)
    num_classes = len(sign_to_idx)
    print(f"Classes: {num_classes}")

    index_path = args.processed / "index.json"
    if not index_path.exists():
        sys.exit(f"index.json not found at {index_path}. Run preprocess_signs.py.")
    with open(index_path) as f:
        index = json.load(f)
    rows = [
        {"sequence_id": int(sid), **v}
        for sid, v in index.items()
    ]
    meta = pd.DataFrame(rows)
    if meta.empty:
        sys.exit("No clips in index.json")

    if args.smoke:
        meta = meta.sample(n=min(512, len(meta)), random_state=42).reset_index(drop=True)
        args.epochs = 2

    # Held-out signers as val.
    rng = np.random.default_rng(42)
    participants = sorted(meta["participant_id"].unique())
    rng.shuffle(participants)
    n_val = max(1, int(0.05 * len(participants)))
    val_ids = set(participants[:n_val])
    train_meta = meta[~meta["participant_id"].isin(val_ids)].reset_index(drop=True)
    val_meta = meta[meta["participant_id"].isin(val_ids)].reset_index(drop=True)
    print(f"Train clips: {len(train_meta)}, val clips: {len(val_meta)} "
          f"({len(val_ids)} held-out signers)")

    train_ds = SignsDataset(
        args.processed, train_meta, max_frames=args.max_frames, augment=args.augment,
    )
    val_ds = SignsDataset(
        args.processed, val_meta, max_frames=args.max_frames, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=signs_collate,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=signs_collate,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    model = build_sign_classifier(
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
    ).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=(DEVICE == "cuda"))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    best_val_acc = 0.0
    history: list[dict] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss = 0.0
        tr_n = 0
        tr_correct = 0
        for batch_idx, batch in enumerate(train_loader, 1):
            if batch is None:
                continue
            x, labels, pad_mask = batch
            x = x.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            pad_mask = pad_mask.to(DEVICE, non_blocking=True)

            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(global_step)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                logits = model(x, pad_mask=pad_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * x.size(0)
            tr_correct += (logits.argmax(-1) == labels).sum().item()
            tr_n += x.size(0)
            global_step += 1
            if batch_idx % 100 == 0:
                print(
                    f"  e{epoch} step {batch_idx}/{steps_per_epoch}  "
                    f"loss={tr_loss / max(tr_n, 1):.4f}  "
                    f"acc={tr_correct / max(tr_n, 1):.3f}  "
                    f"lr={lr_at(global_step):.2e}",
                    flush=True,
                )

        tr_loss /= max(tr_n, 1)
        tr_acc = tr_correct / max(tr_n, 1)

        # Validation
        model.eval()
        vl_loss, vl_n, vl_correct = 0.0, 0, 0
        vl_top5 = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                x, labels, pad_mask = batch
                x = x.to(DEVICE); labels = labels.to(DEVICE); pad_mask = pad_mask.to(DEVICE)
                logits = model(x, pad_mask=pad_mask)
                loss = criterion(logits, labels)
                vl_loss += loss.item() * x.size(0)
                vl_correct += (logits.argmax(-1) == labels).sum().item()
                _, top5 = logits.topk(5, dim=-1)
                vl_top5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
                vl_n += x.size(0)
        vl_loss /= max(vl_n, 1)
        vl_acc = vl_correct / max(vl_n, 1)
        vl_top5_acc = vl_top5 / max(vl_n, 1)

        improved = vl_acc > best_val_acc
        if improved:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), args.exports / "best.pt")

        dt = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": vl_loss,
            "val_acc": vl_acc,
            "val_top5": vl_top5_acc,
            "seconds": dt,
        })
        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {tr_loss:.4f}/{tr_acc:.3f} | "
            f"val {vl_loss:.4f}/{vl_acc:.3f} top5 {vl_top5_acc:.3f}{marker} | "
            f"{dt:.0f}s"
        )

    with open(args.exports / "model_meta.json", "w") as f:
        json.dump({
            "num_classes": num_classes,
            "sign_to_idx": sign_to_idx,
            "idx_to_sign": {str(k): v for k, v in idx_to_sign.items()},
            "d_model": args.d_model,
            "nhead": args.nhead,
            "layers": args.layers,
            "max_frames": args.max_frames,
            "best_val_acc": best_val_acc,
            "history": history,
        }, f, indent=2)

    print(f"\nBest val top-1: {best_val_acc:.3f}")
    print(f"Saved to {args.exports}/")
    print("Next steps:")
    print("  python scripts/export_signs_onnx.py")
    print("  python scripts/build_sign_references.py")


if __name__ == "__main__":
    main()
