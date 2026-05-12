"""
Train the ASL MLP classifier.

Usage:
  python scripts/train.py [--data data/processed] [--epochs 60] [--batch 256]

Checkpoints saved to exports/checkpoints/.
Best model (by val accuracy) saved to exports/best.pt.
Training curves saved to exports/training_curves.png.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.mlp import build_model
from model.dataset import load_splits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            logits = model(X)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def plot_curves(history: dict, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed", type=Path)
    parser.add_argument("--exports", default="exports", type=Path)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--batch", default=256, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--patience", default=10, type=int,
                        help="Early stopping patience (epochs without val improvement)")
    args = parser.parse_args()

    ckpt_dir = args.exports / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading data from {args.data} ...")
    train_ds, val_ds, test_ds = load_splits(args.data)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    with open(args.data / "label_map.json") as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=0)

    model = build_model(num_classes=num_classes).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=(DEVICE == "cuda"))

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        improved = vl_acc > best_val_acc
        if improved:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), args.exports / "best.pt")
            no_improve = 0
        else:
            no_improve += 1

        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {vl_loss:.4f} acc {vl_acc:.3f}{marker}"
        )

        if no_improve >= args.patience:
            print(f"Early stopping after {epoch} epochs (no val improvement for {args.patience} epochs)")
            break

    # Final test evaluation
    model.load_state_dict(torch.load(args.exports / "best.pt", map_location=DEVICE, weights_only=True))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion)
    print(f"\nTest accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")

    # Save training curves
    plot_curves(history, args.exports / "training_curves.png")
    print(f"Training curves saved to {args.exports / 'training_curves.png'}")

    # Save metadata alongside the checkpoint
    meta = {
        "num_classes": num_classes,
        "label_map": label_map,
        "input_dim": 63,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "epochs_trained": len(history["train_loss"]),
    }
    with open(args.exports / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBest checkpoint: {args.exports / 'best.pt'}")
    print("Next step: python scripts/export_onnx.py")


if __name__ == "__main__":
    main()
