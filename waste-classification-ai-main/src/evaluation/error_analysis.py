"""Error analysis utilities: identifying and visualizing misclassified test images.

Kept separate from src/evaluation/metrics.py because that module is about aggregate
metrics (shared across every model); this one is about inspecting individual
predictions, which is inherently tied to a specific evaluated model run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def build_misclassification_table(
    test_manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Builds a DataFrame of every misclassified test image with its confidence and
    top-3 predictions. `test_manifest` must be in the same row order the model was
    evaluated in (i.e. the un-shuffled test split DataFrame).
    """
    records = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            continue
        probs = y_probs[i]
        top3_idx = np.argsort(probs)[::-1][:3]
        records.append(
            {
                "path": test_manifest.iloc[i]["path"],
                "actual": class_names[y_true[i]],
                "predicted": class_names[y_pred[i]],
                "confidence": float(probs[y_pred[i]]),
                "top3": [(class_names[j], float(probs[j])) for j in top3_idx],
            }
        )
    return pd.DataFrame.from_records(records)


def plot_misclassified_examples(
    misclassified_df: pd.DataFrame,
    n_examples: int = 12,
    save_path: str | Path | None = None,
    seed: int = 42,
) -> plt.Figure:
    """Grid of representative misclassified images, each captioned with actual vs.
    predicted class and confidence. Samples are chosen to spread across different
    (actual, predicted) confusion pairs rather than just taking the first N, so the
    grid is representative of the range of errors rather than one dominant pair.
    """
    rng = np.random.default_rng(seed)
    df = misclassified_df.copy()
    df["pair"] = df["actual"] + " -> " + df["predicted"]

    # Sample roughly evenly across distinct confusion pairs, then top up randomly.
    pairs = df["pair"].unique()
    per_pair = max(1, n_examples // max(1, len(pairs)))
    sampled_idx = []
    for pair in pairs:
        pair_idx = df.index[df["pair"] == pair].tolist()
        take = min(per_pair, len(pair_idx))
        sampled_idx.extend(rng.choice(pair_idx, size=take, replace=False).tolist())
    remaining = [i for i in df.index if i not in sampled_idx]
    if len(sampled_idx) < n_examples and remaining:
        extra = rng.choice(remaining, size=min(n_examples - len(sampled_idx), len(remaining)), replace=False)
        sampled_idx.extend(extra.tolist())
    sampled_idx = sampled_idx[:n_examples]

    n_cols = 4
    n_rows = int(np.ceil(len(sampled_idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.4))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, sampled_idx):
        row = df.loc[idx]
        img = Image.open(row["path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"Actual: {row['actual'].capitalize()}\n"
            f"Predicted: {row['predicted'].capitalize()}\n"
            f"Confidence: {row['confidence']*100:.1f}%",
            fontsize=9,
        )
    for ax in axes[len(sampled_idx):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
