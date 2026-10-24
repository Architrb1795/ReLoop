"""Grad-CAM (Gradient-weighted Class Activation Mapping) for the ResNet50 model.

Implemented directly with forward/backward hooks (no extra dependency) since only one
architecture needs it and the technique is simple enough to keep transparent/educational.

## What Grad-CAM shows (and doesn't)

Grad-CAM highlights which spatial regions of the input most influenced the model's score
for a chosen class, by combining the target convolutional layer's activations with the
gradient of that class's score with respect to those activations. It answers "where did the
network's final decision draw evidence from," not "why is this the correct class" or
"what does this class mean" — it is a visualization of contribution, not a proof of
causality, and it reflects the *model's* learned associations, which can include spurious
correlations the model discovered rather than genuinely meaningful features.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM for a CNN with a target convolutional layer.

    For ResNet50, the target layer is `model.layer4[-1]` — the last residual block, right
    before global average pooling. This is the standard choice: it's the deepest layer that
    still retains spatial structure (7x7 for a 224x224 input), so its activations correspond
    to "what was seen where," while being close enough to the output that its gradients
    reflect the final classification decision.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int, float]:
        """Runs a forward+backward pass and returns (cam, predicted_class, confidence).

        `input_tensor` must be a single preprocessed image, shape (1, 3, H, W), with
        gradients enabled on the graph (no `torch.no_grad()` around the call).
        `cam` is a (H, W) array in [0, 1], already resized to match the input resolution.
        If `target_class` is None, uses the model's own top prediction (explaining "why did
        the model predict what it predicted"); pass an explicit class index to instead see
        evidence for a *different* class (e.g. "why didn't it predict the true label").
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        confidence = float(probs[0, target_class].item())

        score = output[0, target_class]
        score.backward()

        # Global-average-pool the gradients over spatial dims -> one importance weight per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # only positive contributions to the target class

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam, target_class, confidence


def overlay_heatmap(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlays a Grad-CAM heatmap (H, W) in [0,1] onto the original PIL image.
    Returns an (H, W, 3) uint8 array of the blended overlay.
    """
    resized = original_image.resize((cam.shape[1], cam.shape[0]))
    img_arr = np.array(resized).astype(float) / 255.0

    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[:, :, :3]  # drop alpha channel from colormap

    overlay = (1 - alpha) * img_arr + alpha * heatmap
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def plot_gradcam_panel(
    image_path: str,
    cam: np.ndarray,
    actual_class: str,
    predicted_class: str,
    confidence: float,
    save_path: str | Path,
) -> None:
    """Saves a 3-panel figure: original | heatmap | overlay, captioned with actual/predicted/confidence."""
    original = Image.open(image_path).convert("RGB")
    overlay = overlay_heatmap(original, cam)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    correct = actual_class == predicted_class
    status = "CORRECT" if correct else "INCORRECT"
    fig.suptitle(
        f"Actual: {actual_class.capitalize()}  |  Predicted: {predicted_class.capitalize()} "
        f"({confidence*100:.1f}%)  —  {status}",
        fontsize=11,
        color="darkgreen" if correct else "darkred",
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
