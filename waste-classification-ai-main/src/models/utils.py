"""Shared model utilities — used by both the from-scratch CNN and every
transfer-learning model so parameter reporting is consistent across Phase 3
and Phase 4.
"""
from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable, "frozen_params": total - trainable}
