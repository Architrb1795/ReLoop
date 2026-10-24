"""Image transform pipelines for training vs. evaluation.

Two separate pipelines are defined because they serve different purposes:

- **Training transforms** include randomized data augmentation. Every epoch,
  the model sees a slightly different version of each image (flipped,
  rotated, recolored, cropped). This artificially grows the effective size
  and diversity of a small dataset (2,527 images) and makes it harder for
  the model to memorize exact pixels, which reduces overfitting.

- **Validation/test transforms are deterministic** — resize + normalize only,
  no randomness. We need validation and test metrics to be reproducible and
  comparable across epochs/models; if they included random augmentation, a
  score could swing between runs for reasons that have nothing to do with
  the model improving, and we'd have no reliable evaluation signal.

All transforms end in `ToTensor()` + `Normalize(IMAGENET_MEAN, IMAGENET_STD)`
so every model (from-scratch CNN and every pretrained backbone) trains on
identically scaled inputs.
"""
from __future__ import annotations

from torchvision import transforms

from src.utils.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Training pipeline: randomized augmentation + normalization.

    Each transform and why it's included:

    - `RandomResizedCrop(img_size, scale=(0.8, 1.0))`: crops a random
      (80-100%-area) region of the image and resizes it to `img_size`.
      Simulates the object appearing at a different scale/position than
      dead-center, since TrashNet photos are consistently centered — without
      this, the model could learn "the object is in the middle" as a shortcut.
    - `RandomHorizontalFlip()`: waste items have no inherent left/right
      orientation, so flipping doesn't change the true label — free, valid
      augmentation.
    - `RandomRotation(15)`: photos aren't taken at a perfectly consistent
      angle in the real world. A small rotation (±15°) teaches tolerance to
      that without distorting the object beyond recognition.
    - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`: varies
      lighting/exposure. TrashNet photos share fairly uniform studio
      lighting; real-world deployment will not, so this narrows that gap.
    - `ToTensor()`: converts a PIL image (H, W, C) in [0, 255] to a
      `torch.Tensor` (C, H, W) in [0.0, 1.0] — the format PyTorch models
      require.
    - `Normalize(IMAGENET_MEAN, IMAGENET_STD)`: rescales each channel to
      roughly zero mean / unit variance, using the same statistics the
      pretrained ImageNet backbones were trained with. This keeps input
      distributions in the range those pretrained weights expect and helps
      the from-scratch CNN's gradients stay well-scaled too.

    Note what's deliberately *not* here: `RandomResizedCrop` already resizes,
    so no separate `Resize` step is needed for training.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Validation/test pipeline: deterministic resize + normalization only.

    - `Resize((img_size, img_size))`: deterministic resize to the model's
      expected input size — no cropping/randomness, so the same image always
      produces the same tensor.
    - `ToTensor()` + `Normalize(...)`: identical to training, so the model
      sees the same input distribution it was trained on.

    Deliberately excludes every augmentation above: validation/test scores
    must measure the model's true generalization on realistic, unmodified
    inputs, and must be reproducible run-to-run for meaningful comparison
    across epochs and between models.
    """
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
