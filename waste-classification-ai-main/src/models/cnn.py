"""A CNN built manually (no pretrained weights) — the Phase 3 baseline that
transfer-learning models (Phase 4) will be compared against.

Architecture (input 3x224x224):

    [ConvBlock  3 ->  32]  224x224 -> 112x112
    [ConvBlock 32 ->  64]  112x112 ->  56x56
    [ConvBlock 64 -> 128]   56x56  ->  28x28
    [ConvBlock 128-> 256]   28x28  ->  14x14
    GlobalAveragePool       14x14  ->  1x1   (256 channels)
    Dropout(p=0.5)
    Linear(256 -> num_classes)

Each ConvBlock is Conv2d(k=3, pad=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2).

Design choices, and why:

- **4 conv blocks, not 3.** The project brief's example architecture has 3
  blocks, but that assumes a smaller input. We standardized on 224x224 input
  (see `src/utils/config.py`) so the from-scratch CNN and every pretrained
  transfer-learning backbone see identically sized/preprocessed images —
  otherwise "model comparison" (Phase 6) wouldn't be a fair comparison. At
  224x224, 3 pooling stages only gets down to 28x28 (28*28*128 ≈ 100k
  activations feeding the classifier) — a 4th block brings that down to a
  much more manageable 14x14 before global pooling, cutting both compute and
  overfitting risk.
- **3x3 kernels throughout**, the standard choice since VGG: stacking small
  kernels gets the same receptive field as one large kernel with fewer
  parameters and more nonlinearities in between (more expressive per
  parameter).
- **BatchNorm after every conv, before activation.** Normalizes each layer's
  input distribution during training, which lets us use a higher learning
  rate and train faster/more stably. It also has a mild regularizing effect
  since batch statistics add noise similar to (but weaker than) dropout.
  Without it, this network would likely need a much lower learning rate and
  more epochs to converge, and would be more sensitive to initialization.
- **ReLU** activation: cheap, avoids the vanishing-gradient problem that
  sigmoid/tanh have for deeper stacks, and is the standard default for CNNs.
- **MaxPool2d(2) after every block**: halves spatial resolution, keeps the
  strongest activation in each 2x2 window (the most "present" feature),
  reduces compute for every subsequent layer, and adds a small amount of
  translation invariance (a feature shifted by 1px still likely survives
  pooling to the same output).
- **Global Average Pooling instead of Flatten + large Dense layer.** Flattening
  the 14x14x256 feature map would need a `Linear(50176, num_classes)` layer —
  over 300K parameters in one layer, prone to overfitting on a dataset this
  small (1,768 training images). GAP instead averages each of the 256
  channels down to a single number, giving a `Linear(256, num_classes)`
  classifier — a fraction of the parameters, and it forces each channel to
  represent "how much of this feature is present anywhere in the image"
  rather than depending on exact spatial position.
- **Dropout(p=0.5) before the final layer.** Randomly zeroes half the
  pooled features each training step, which prevents the classifier from
  co-adapting to specific feature combinations — again, important
  regularization given the small dataset. Removing it would likely show up
  as a widening gap between train and validation accuracy (overfitting).

What happens if key pieces are removed (for the "what could happen if we
remove it" teaching requirement):
- **No BatchNorm:** slower, less stable convergence; more sensitive to
  learning rate and weight initialization.
- **No Dropout:** the model would likely fit the training set increasingly
  well while validation accuracy plateaus or degrades — classic overfitting,
  worse on a dataset this small.
- **No GAP (flatten instead):** far more parameters, higher overfitting risk,
  larger model file size, slower training — with a 2.5K-image dataset this
  would likely hurt validation performance rather than help it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.utils import count_parameters  # noqa: F401  (re-exported for backward compat)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class WasteCNN(nn.Module):
    """From-scratch CNN for 6-class waste classification. Input: 3x224x224."""

    def __init__(self, num_classes: int = 6, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.classifier(x)
