"""
ReLoop — Unified Computer Vision Inference, Open-Set OOD Rejection & RVM Engine
===============================================================================
Uses the trained and validation-calibrated ResNet50 model (`reloop_resnet50_calibrated.pt`
or `resnet50_best.pt`) from `waste-classification-ai-main` for BOTH uploaded images
and live camera frames.

Implements a 4-stage inference & verification pipeline:
  1. Image Validity & Quality Gate (`POOR_QUALITY` for dark, overexposed, blurry, or empty frames).
  2. Multi-View ResNet50 Deep Feature & Logit Extraction (224x224 ImageNet-normalized).
  3. Open-Set / Out-of-Distribution (OOD) & Stone/Non-Waste Rejection (`UNKNOWN`) using
     the 18-dimensional Open-Set Discriminator (2048-d Class-Conditional Mahalanobis
     distance, Prototype Cosine Similarity, Calibrated Logit Energy, and 12-d Physical
     Material & Texture Descriptors).
  4. Calibrated Confidence & Top-2 Margin Check (`UNCERTAIN` vs `RECOGNIZED`).
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1] / "waste-classification-ai-main"
sys.path.insert(0, str(REPO_ROOT))

import torch
from torchvision import transforms

from src.models.transfer import build_transfer_model
from src.training.upgrade_ood_detector import extract_12d_material_descriptors
from src.utils.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, MODELS_DIR
from src.utils.device import get_device

torch.set_num_threads(max(4, os.cpu_count() or 4))

RELOOP_CLASS_CONFIG: dict[str, dict[str, Any]] = {
    "plastic": {
        "display_name": "Plastic Bottle / Container",
        "short_label": "Plastic",
        "stream": "Polymer Recycling Stream (PET/HDPE)",
        "bin_label": "Recyclable — Plastics",
        "servo_angle": 120,
        "points": 20,
        "est_weight_kg": 0.025,
        "recovery_factor": 0.88,
        "guidance": "Empty liquid residue before depositing into the RVM intake.",
    },
    "paper": {
        "display_name": "Paper Cup / Sheet",
        "short_label": "Paper",
        "stream": "Clean Pulp & Fiber Recovery",
        "bin_label": "Recyclable — Paper",
        "servo_angle": 120,
        "points": 10,
        "est_weight_kg": 0.015,
        "recovery_factor": 0.82,
        "guidance": "Ensure paper is free from heavy food grease.",
    },
    "metal": {
        "display_name": "Aluminium Can / Metal Tin",
        "short_label": "Aluminium / Metal",
        "stream": "High-Value Metal Smelting Recovery",
        "bin_label": "Recyclable — Metals",
        "servo_angle": 120,
        "points": 30,
        "est_weight_kg": 0.018,
        "recovery_factor": 0.96,
        "guidance": "High-recovery material. Deposit directly into intake chamber.",
    },
    "glass": {
        "display_name": "Glass Bottle / Jar",
        "short_label": "Glass",
        "stream": "Cullet Glass Remelting Stream",
        "bin_label": "Recyclable — Glass",
        "servo_angle": 120,
        "points": 25,
        "est_weight_kg": 0.180,
        "recovery_factor": 0.95,
        "guidance": "Place intact glass container gently on the intake platform.",
    },
    "cardboard": {
        "display_name": "Corrugated Cardboard Packaging",
        "short_label": "Cardboard",
        "stream": "Corrugated Fiberboard Recovery",
        "bin_label": "Recyclable — Cardboard",
        "servo_angle": 120,
        "points": 15,
        "est_weight_kg": 0.065,
        "recovery_factor": 0.90,
        "guidance": "Flatten small boxes if needed to clear the optical chamber.",
    },
    "trash": {
        "display_name": "Mixed / Non-Recyclable Wrapper",
        "short_label": "Residual Waste",
        "stream": "Controlled Non-Recyclable Containment",
        "bin_label": "Non-Recyclable Compartment",
        "servo_angle": 180,
        "points": 5,
        "est_weight_kg": 0.012,
        "recovery_factor": 0.0,
        "guidance": "Routed to isolated residual bin to prevent recyclable contamination.",
    },
}


class ReLoopVisionEngine:
    """Unified production inference engine for Uploaded Images and Live Camera frames."""

    def __init__(self):
        self.device = get_device()
        self.class_names = list(CLASS_NAMES)
        self.calibrated_bundle_path = MODELS_DIR / "reloop_resnet50_calibrated.pt"
        self.baseline_ckpt_path = MODELS_DIR / "resnet50_best.pt"
        self.active_checkpoint_name = ""

        self.model = build_transfer_model(
            "resnet50", num_classes=len(self.class_names), pretrained=False
        )
        self.clf_weights: Optional[np.ndarray] = None
        self.clf_intercept: Optional[np.ndarray] = None
        self.prototypes: Optional[np.ndarray] = None
        self.ood_detector: Optional[dict[str, Any]] = None
        self.calibration_params = {
            "alpha_head": 0.70,
            "beta_proto": 0.0,
            "trash_bias_correction": -0.35,
            "temperature": 0.85,
        }
        self.knownness_tau = 0.50
        self.uncertainty_tau = 0.58

        self._load_weights()
        self.views = self._build_two_view_transforms()

    def _load_weights(self) -> None:
        if self.calibrated_bundle_path.exists():
            bundle = torch.load(
                self.calibrated_bundle_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(bundle["backbone_state_dict"])
            self.clf_weights = bundle["clf_weights"].cpu().numpy()
            self.clf_intercept = bundle["clf_intercept"].cpu().numpy()
            self.prototypes = bundle["prototypes"].cpu().numpy()
            self.calibration_params = bundle["calibration_params"]
            self.ood_detector = bundle.get("ood_detector")
            self.knownness_tau = float(bundle["ood_thresholds"]["knownness_tau"])
            self.uncertainty_tau = float(bundle["ood_thresholds"]["uncertainty_tau"])
            self.active_checkpoint_name = "models/reloop_resnet50_calibrated.pt"
        elif self.baseline_ckpt_path.exists():
            state_dict = torch.load(
                self.baseline_ckpt_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(state_dict)
            self.active_checkpoint_name = "models/resnet50_best.pt"
        else:
            raise FileNotFoundError(
                f"Neither {self.calibrated_bundle_path} nor {self.baseline_ckpt_path} exists."
            )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _build_two_view_transforms():
        norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        v0 = transforms.Compose(
            [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm]
        )
        v1 = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                norm,
            ]
        )
        return [v0, v1]

    @staticmethod
    def assess_image_quality(image: Image.Image, mat_desc: np.ndarray) -> dict[str, Any]:
        """Checks resolution, darkness, glare, blur, and empty uniform background."""
        w, h = image.size
        if w < 32 or h < 32:
            return {
                "quality_ok": False,
                "reason": "Image resolution is too small (minimum 32x32 pixels required).",
                "code": "RESOLUTION_TOO_LOW",
            }

        brightness_mean = float(mat_desc[0] * 255.0)
        contrast_std = float(mat_desc[1] * 128.0)
        lap_var = float(np.expm1(mat_desc[7] * 10.0))

        if brightness_mean < 18.0:
            return {
                "quality_ok": False,
                "reason": "Image is too dark. Please increase lighting or move closer to the camera.",
                "code": "TOO_DARK",
            }

        if brightness_mean > 247.0 and contrast_std < 7.0:
            return {
                "quality_ok": False,
                "reason": "Image is overexposed or blank white. Please avoid direct glare.",
                "code": "OVEREXPOSED_OR_BLANK",
            }

        if contrast_std < 6.5:
            return {
                "quality_ok": False,
                "reason": "No distinct object detected in the frame (uniform background).",
                "code": "EMPTY_BACKGROUND",
            }

        if lap_var < 10.5 and contrast_std < 18.0:
            return {
                "quality_ok": False,
                "reason": "Image is too blurry or out of focus. Hold the item steady and retry.",
                "code": "TOO_BLURRY",
            }

        return {"quality_ok": True}

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        source: str = "upload",
        filename_hint: str = "",
    ) -> dict[str, Any]:
        """Unified prediction method for both uploaded images and live camera frames."""
        t0 = time.perf_counter()
        rgb = image.convert("RGB")
        mat_desc = extract_12d_material_descriptors(rgb)

        # Step 1: Image Quality Check
        quality = self.assess_image_quality(rgb, mat_desc)
        if not quality["quality_ok"]:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "prediction_id": f"pred_{uuid.uuid4().hex[:10]}",
                "status": "POOR_QUALITY",
                "recognized": False,
                "acceptable_for_segregation": False,
                "predicted_class": None,
                "display_label": "Poor Image Quality — Retry Required",
                "confidence": 0.0,
                "knownness_score": 0.0,
                "reason": quality["reason"],
                "quality_code": quality["code"],
                "sorting_action": "HOLD_NO_GATE_ACTUATION",
                "reward_points_eligible": 0,
                "top_predictions": [],
                "latency_ms": round(latency_ms, 1),
                "model_checkpoint": self.active_checkpoint_name,
                "source": source,
                "is_simulated_hardware": True,
            }

        # Step 2: 2-View Deep Feature & Logit Extraction
        captured_feats = []

        def _hook(module, inp, out):
            captured_feats.append(out.flatten(1).detach().cpu())

        handle = self.model.avgpool.register_forward_hook(_hook)
        tensors = torch.stack([tfm(rgb) for tfm in self.views]).to(self.device)
        logits_t = self.model(tensors).cpu().numpy()
        feats_t = torch.cat(captured_feats, dim=0).numpy()
        handle.remove()

        base_logits = 0.60 * logits_t[0] + 0.40 * logits_t[1]
        feats_mv = 0.60 * feats_t[0] + 0.40 * feats_t[1]
        feats_norm = feats_mv / (np.linalg.norm(feats_mv) + 1e-12)

        # Step 3: Calibrated Hybrid Classifier Head
        if self.clf_weights is not None and self.prototypes is not None:
            head_logits = feats_norm @ self.clf_weights.T + self.clf_intercept
            proto_sims = feats_norm @ self.prototypes.T
            p = self.calibration_params
            comb_logits = (
                (1.0 - p["alpha_head"]) * base_logits
                + p["alpha_head"] * head_logits
                + p["beta_proto"] * proto_sims
            )
            comb_logits[5] += p["trash_bias_correction"]
            temp = float(p["temperature"])
            sorted_sims = np.sort(proto_sims)
            max_proto_sim = float(sorted_sims[-1])
            second_proto_sim = float(sorted_sims[-2])
        else:
            comb_logits = base_logits
            temp = 1.0
            max_proto_sim = 0.68
            second_proto_sim = 0.45

        scaled = (comb_logits / temp) - np.max(comb_logits / temp)
        probs = np.exp(scaled) / np.sum(np.exp(scaled))

        ranked_indices = np.argsort(-probs)
        top1_idx = int(ranked_indices[0])
        top2_idx = int(ranked_indices[1])
        top1_class = self.class_names[top1_idx]
        top2_class = self.class_names[top2_idx]
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[top2_idx])
        margin = top1_prob - top2_prob
        energy = float(temp * np.log(np.sum(np.exp(comb_logits / temp))))

        # Step 4: Compute Open-Set Knownness Probability via 18-d Discriminator
        if self.ood_detector is not None:
            od = self.ood_detector
            diff = feats_norm - od["cls_means"][top1_idx]
            maha_dist = -float(np.mean((diff * diff) / od["cls_vars"][top1_idx]))
            norm_energy = float(np.tanh(energy / 10.0))
            base_6 = np.array(
                [max_proto_sim, second_proto_sim, top1_prob, margin, norm_energy, maha_dist],
                dtype=np.float32,
            )
            x_18 = np.concatenate([base_6, mat_desc], axis=0)
            x_std = (x_18 - od["feat_mean"]) / od["feat_std"]
            logit_os = float((x_std @ od["ood_coef"].T + od["ood_intercept"]).item())
            knownness_score = float(1.0 / (1.0 + np.exp(-np.clip(logit_os, -25.0, 25.0))))
        else:
            knownness_score = (
                0.50 * max_proto_sim
                + 0.30 * top1_prob
                + 0.12 * margin
                + 0.08 * float(np.tanh(energy / 10.0))
            )

        top_predictions = [
            {
                "class": self.class_names[int(i)],
                "label": RELOOP_CLASS_CONFIG[self.class_names[int(i)]]["short_label"],
                "probability": round(float(probs[int(i)]), 4),
            }
            for i in ranked_indices[:3]
        ]

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Step 5: Open-Set / Out-of-Distribution (Unknown Object) Rejection Decision
        if knownness_score < self.knownness_tau or max_proto_sim < 0.42:
            return {
                "prediction_id": f"pred_{uuid.uuid4().hex[:10]}",
                "status": "UNKNOWN",
                "recognized": False,
                "acceptable_for_segregation": False,
                "predicted_class": None,
                "display_label": "Unknown Object — Cannot Reliably Identify",
                "confidence": round(top1_prob, 4),
                "knownness_score": round(float(knownness_score), 4),
                "prototype_similarity": round(max_proto_sim, 4),
                "reason": (
                    "This object falls outside ReLoop's supported waste categories "
                    "(Cardboard, Glass, Metal, Paper, Plastic, Residual Wrapper). "
                    "Automatic sorting gate locked; routed to Safe Reject Compartment."
                ),
                "sorting_action": "SAFE_REJECT_COMPARTMENT",
                "reward_points_eligible": 0,
                "top_predictions": top_predictions,
                "latency_ms": round(latency_ms, 1),
                "model_checkpoint": self.active_checkpoint_name,
                "source": source,
                "is_simulated_hardware": True,
            }

        # Step 6: Uncertainty Check (Ambiguous between two known waste classes)
        if (
            knownness_score < self.uncertainty_tau
            or top1_prob < 0.56
            or (top1_prob < 0.70 and margin < 0.14)
        ):
            c1_lbl = RELOOP_CLASS_CONFIG[top1_class]["short_label"]
            c2_lbl = RELOOP_CLASS_CONFIG[top2_class]["short_label"]
            return {
                "prediction_id": f"pred_{uuid.uuid4().hex[:10]}",
                "status": "UNCERTAIN",
                "recognized": False,
                "acceptable_for_segregation": False,
                "predicted_class": None,
                "display_label": f"Uncertain Prediction ({c1_lbl} vs. {c2_lbl})",
                "confidence": round(top1_prob, 4),
                "knownness_score": round(float(knownness_score), 4),
                "prototype_similarity": round(max_proto_sim, 4),
                "reason": (
                    f"Model confidence ({top1_prob*100:.1f}%) is below the safe automatic "
                    f"segregation threshold due to visual overlap between {c1_lbl} and {c2_lbl}. "
                    f"Please reposition the item or provide a clearer angle."
                ),
                "sorting_action": "HOLD_FOR_REPOSITION_OR_MANUAL_REVIEW",
                "reward_points_eligible": 0,
                "top_predictions": top_predictions,
                "latency_ms": round(latency_ms, 1),
                "model_checkpoint": self.active_checkpoint_name,
                "source": source,
                "is_simulated_hardware": True,
            }

        # Step 7: Confidently Recognized Supported Waste Class
        cfg = RELOOP_CLASS_CONFIG[top1_class]
        return {
            "prediction_id": f"pred_{uuid.uuid4().hex[:10]}",
            "status": "RECOGNIZED",
            "recognized": True,
            "acceptable_for_segregation": True,
            "predicted_class": top1_class,
            "display_label": cfg["display_name"],
            "short_label": cfg["short_label"],
            "confidence": round(top1_prob, 4),
            "knownness_score": round(float(knownness_score), 4),
            "prototype_similarity": round(max_proto_sim, 4),
            "sorting_stream": cfg["stream"],
            "bin_label": cfg["bin_label"],
            "servo_angle_simulated": cfg["servo_angle"],
            "sorting_action": f"SIMULATED_ROUTE_TO_{top1_class.upper()}_BIN",
            "reward_points_eligible": cfg["points"],
            "estimated_weight_kg": cfg["est_weight_kg"],
            "estimated_recovered_kg": round(cfg["est_weight_kg"] * cfg["recovery_factor"], 4),
            "disposal_guidance": cfg["guidance"],
            "reason": f"Verified {cfg['short_label']} with {top1_prob*100:.1f}% calibrated confidence.",
            "top_predictions": top_predictions,
            "latency_ms": round(latency_ms, 1),
            "model_checkpoint": self.active_checkpoint_name,
            "source": source,
            "is_simulated_hardware": True,
        }
