"""Multi-epoch training loop with validation tracking, best-model
checkpointing, and early stopping.

Early stopping and checkpointing both watch **validation loss**, not
training loss — training loss will keep improving even as the model starts
overfitting, so it's not a useful signal for "when to stop" or "which
epoch's weights to keep."
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.engine import evaluate, train_one_epoch


class EarlyStopping:
    """Stops training when validation loss hasn't improved for `patience` epochs.

    Why early stopping: without it, a model will keep training past the point
    of diminishing (or negative) returns on validation performance, wasting
    compute and typically ending up more overfit than the best checkpoint
    along the way. `min_delta` guards against stopping on noise — an
    improvement smaller than `min_delta` doesn't reset the patience counter.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
    patience: int = 5,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> dict:
    """Trains `model` for up to `epochs` epochs, with early stopping and
    best-checkpoint saving. Returns a history dict with per-epoch metrics,
    suitable for both plotting and JSON serialization.
    """
    model.to(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(patience=patience)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time_sec": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - start
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time_sec"].append(elapsed)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
            f"{'  [best, checkpoint saved]' if improved else ''}"
        )

        if early_stopping.step(val_loss):
            print(f"Early stopping triggered after epoch {epoch} (no improvement for {patience} epochs).")
            break

    return history


def save_history(history: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
