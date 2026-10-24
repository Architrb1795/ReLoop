"""Transfer-learning model builders and freeze/fine-tune logic for
MobileNetV2, EfficientNet-B0, and ResNet50 — all pretrained on ImageNet.

## Why transfer learning

TrashNet has only 2,527 images. A CNN trained from scratch (Phase 3) has to
learn everything — edges, textures, shapes, object parts — from that small
amount of data alone, which caps how good it can get and how well it
generalizes. These three architectures were already trained on ImageNet
(1.28M images, 1000 classes), so their early/middle layers already encode
general-purpose visual features (edge detectors, textures, shapes) that
transfer to almost any image classification task, waste included. We reuse
those features instead of relearning them, which should let a much smaller
amount of task-specific data reach higher accuracy, faster, than training
from scratch.

## Two-stage strategy (Stage A / Stage B)

- **Stage A — Feature extraction.** Freeze every backbone weight; only the
  newly-initialized classifier head (which we resize from ImageNet's 1000
  classes to our 6) is trainable. This lets the random classifier weights
  settle into something reasonable without corrupting the pretrained
  features via large, noisy early gradients.
- **Stage B — Fine-tuning.** Unfreeze the backbone's last few
  blocks/layers (the most task-specific ones — early layers stay frozen
  since they encode generic, already-useful low-level features) and
  continue training with a **much lower learning rate**
  (`STAGE_B_LR` = 1e-5 vs. `STAGE_A_LR` = 1e-3 in `src/utils/config.py`).

**Why fine-tuning needs a lower learning rate:** the unfrozen backbone
weights already encode useful, well-trained representations; a large
learning rate would apply large gradient updates that overwrite/destroy that
prior knowledge ("catastrophic forgetting") before the model can benefit
from it, especially given our small dataset can produce noisy gradients. A
small learning rate makes small, careful adjustments — nudging the
pretrained features to better fit waste images specifically, rather than
overwriting them with what a few thousand images alone could teach.

## Architecture notes

- **MobileNetV2**: depthwise-separable convolutions, designed for mobile/edge
  deployment — smallest and fastest of the three, good baseline for "what if
  the Streamlit app needs to run cheaply."
- **EfficientNet-B0**: compound-scaled (depth/width/resolution jointly
  scaled), generally strong accuracy-per-parameter trade-off.
- **ResNet50** (optional, deeper/larger): residual connections let much
  deeper networks train without vanishing gradients; typically strong
  accuracy at higher parameter/compute cost than the other two.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models

from src.models.utils import count_parameters  # noqa: F401  (re-exported for convenience)

ARCHITECTURES = ["mobilenet_v2", "efficientnet_b0", "resnet50"]


def build_transfer_model(architecture: str, num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """Builds one of the three supported architectures with ImageNet weights
    and a classifier head resized to `num_classes`. The backbone starts
    fully trainable — call `freeze_backbone()` for Stage A.
    """
    if architecture == "mobilenet_v2":
        weights = tv_models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif architecture == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif architecture == "resnet50":
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown architecture {architecture!r}. Expected one of {ARCHITECTURES}.")

    return model


def freeze_backbone(model: nn.Module, architecture: str) -> None:
    """Stage A: freeze every parameter except the new classifier head.

    The "head" parameter-name prefix differs per architecture
    (`classifier` for MobileNetV2/EfficientNet-B0, `fc` for ResNet50)
    because torchvision names the final layer differently per family.
    """
    head_prefix = {"mobilenet_v2": "classifier", "efficientnet_b0": "classifier", "resnet50": "fc"}[architecture]
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(head_prefix)


def unfreeze_top_layers(model: nn.Module, architecture: str, num_blocks: int = 2) -> None:
    """Stage B: additionally unfreeze the backbone's last `num_blocks`.

    Must be called after `freeze_backbone()` — it only adds trainable
    parameters on top of the frozen state, it doesn't reset anything.

    - MobileNetV2 / EfficientNet-B0: `model.features` is a flat
      `Sequential` of blocks (19 for MobileNetV2, 9 for EfficientNet-B0);
      unfreezes the last `num_blocks` elements.
    - ResNet50: has named stages (`layer1`..`layer4`) rather than a flat
      list, so `num_blocks` selects from the end of
      `[layer1, layer2, layer3, layer4]` — e.g. `num_blocks=1` unfreezes
      only `layer4`, the deepest/most task-specific residual stage.
    """
    if architecture in ("mobilenet_v2", "efficientnet_b0"):
        for block in model.features[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    elif architecture == "resnet50":
        resnet_stages = [model.layer1, model.layer2, model.layer3, model.layer4]
        for stage in resnet_stages[-num_blocks:]:
            for param in stage.parameters():
                param.requires_grad = True

    else:
        raise ValueError(f"Unknown architecture {architecture!r}. Expected one of {ARCHITECTURES}.")
