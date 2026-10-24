"""Class-weighted loss support — addresses the class imbalance found in the
Phase 1 EDA (paper=594 vs. trash=137 images, a 4.3x imbalance).
"""
from __future__ import annotations

import pandas as pd
import torch

from src.utils.config import CLASS_NAMES


def compute_class_weights(manifest: pd.DataFrame, split: str = "train") -> torch.Tensor:
    """Inverse-frequency class weights for `nn.CrossEntropyLoss(weight=...)`.

    weight_c = total_samples / (num_classes * count_c)

    A class with fewer training images gets a proportionally larger weight,
    so a mistake on that class contributes more to the loss. Without this,
    the loss is dominated by the frequent classes (paper, glass, plastic),
    and the model has little incentive to learn `trash`'s weaker, more
    varied signal — it can reach decent overall loss while doing poorly on
    the minority class specifically.
    """
    counts = manifest[manifest["split"] == split]["class"].value_counts().reindex(CLASS_NAMES)
    total = counts.sum()
    weights = total / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights.values, dtype=torch.float32)
