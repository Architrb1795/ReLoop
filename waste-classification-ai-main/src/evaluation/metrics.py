"""Evaluation utilities shared across every model (from-scratch CNN and every
transfer-learning backbone), so the same evaluation code produces directly
comparable results (Phase 6 — Model Comparison).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


@torch.no_grad()
def get_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs `model` over every batch in `loader`. Returns (y_true, y_pred, y_probs).

    `y_probs` holds the full softmax distribution per sample (not just the
    argmax) so callers can report confidence and top-k predictions, not just
    the hard label.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> str:
    return classification_report(y_true, y_pred, target_names=class_names, digits=3)


def get_classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    return classification_report(y_true, y_pred, target_names=class_names, digits=3, output_dict=True)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots a confusion matrix. Normalized (row-wise, i.e. recall per true
    class) by default — with an imbalanced dataset, raw counts make the
    minority class (`trash`) visually negligible even if the model does
    poorly on it. Normalizing per true-class row keeps every class equally
    visible regardless of how many test images it has.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion of true class" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    history: dict, title_prefix: str = "", save_path: str | Path | None = None
) -> plt.Figure:
    """Plots train vs. val loss and train vs. val accuracy side by side.

    Reading these plots (documented here since the project's teaching-mode
    requirement asks for interpretation, not just code):
    - **Both loss curves falling, tracking closely** -> good convergence.
    - **Train loss keeps falling while val loss flattens or rises** ->
      overfitting — the model is fitting training-set noise/specifics that
      don't generalize. Address with more augmentation, dropout, or early
      stopping (which `src/training/trainer.py` already applies).
    - **Both losses stay high / accuracy stays near random-chance (1/6 ≈ 17%
      for this 6-class problem)** -> underfitting — the model doesn't have
      enough capacity, training ran too briefly, or the learning rate is
      poorly tuned.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    axes[0].set_title(f"{title_prefix}Loss".strip())
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="Val Accuracy")
    axes[1].set_title(f"{title_prefix}Accuracy".strip())
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
