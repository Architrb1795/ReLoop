"""
Upgrades and validates the Open-Set / Out-of-Distribution (OOD) & Stone Rejection
mechanism on top of the calibrated ReLoop ResNet50 model, and caches extracted
representations so evaluation and threshold experiments run in seconds.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import WasteDataset
from src.models.transfer import build_transfer_model
from src.training.reloop_pipeline import (
    audit_dataset_and_splits,
    compute_metrics_dict,
    generate_ood_dataset,
    predict_with_calibrated_head,
    softmax_np,
    train_and_tune_improved_classifier,
)
from src.utils.config import (
    CLASS_NAMES,
     DATA_PROCESSED_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.utils.device import get_device
from src.utils.seed import set_seed

torch.set_num_threads(max(4, os.cpu_count() or 4))


def extract_12d_material_descriptors(img: Image.Image) -> np.ndarray:
    """Computes a 12-dimensional physical material & image-quality descriptor vector:
      0: mean_luma / 255
      1: std_luma / 128
      2: center_mean_luma / 255
      3: center_std_luma / 128
      4: specular_tail (p99 - p90 of center luma) / 100
      5: center_mean_saturation
      6: center_std_saturation
      7: log1p(laplacian_var) / 10
      8: log1p(center_laplacian_var) / 10
      9: speckle_to_gradient_ratio
     10: rg_diff_std (color channel variation)
     11: by_diff_std (blue-yellow channel variation)
    """
    rgb = img.convert("RGB").resize((224, 224))
    arr = np.asarray(rgb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    mean_luma = float(np.mean(gray)) / 255.0
    std_luma = float(np.std(gray)) / 128.0

    c_gray = gray[48:176, 48:176]
    c_mean = float(np.mean(c_gray)) / 255.0
    c_std = float(np.std(c_gray)) / 128.0
    p99 = float(np.percentile(c_gray, 99))
    p90 = float(np.percentile(c_gray, 90))
    spec_tail = (p99 - p90) / 100.0

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_c > 5.0, (max_c - min_c) / (max_c + 1e-5), 0.0)
    c_sat = sat[48:176, 48:176]
    c_sat_mean = float(np.mean(c_sat))
    c_sat_std = float(np.std(c_sat))

    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    lap_var = float(np.log1p(np.var(lap))) / 10.0
    c_lap_var = float(np.log1p(np.var(lap[47:175, 47:175]))) / 10.0

    gx = np.abs(gray[48:176, 49:177] - gray[48:176, 47:175])
    gy = np.abs(gray[49:177, 48:176] - gray[47:175, 48:176])
    grad_mag = gx + gy
    p95_grad = float(np.percentile(grad_mag, 95)) + 1e-5
    speckle_ratio = float(np.mean(grad_mag) / p95_grad)

    rg_std = float(np.std(r[48:176, 48:176] - g[48:176, 48:176])) / 128.0
    by_std = float(np.std(0.5 * (r[48:176, 48:176] + g[48:176, 48:176]) - b[48:176, 48:176])) / 128.0

    return np.array(
        [
            mean_luma,
            std_luma,
            c_mean,
            c_std,
            spec_tail,
            c_sat_mean,
            c_sat_std,
            lap_var,
            c_lap_var,
            speckle_ratio,
            rg_std,
            by_std,
        ],
        dtype=np.float32,
    )


def get_two_view_transforms(img_size: int = IMG_SIZE):
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    v0 = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), norm])
    v1 = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            norm,
        ]
    )
    return [v0, v1]


@torch.no_grad()
def load_or_extract_all_features(model: torch.nn.Module, device: torch.device) -> dict:
    cache_path = DATA_PROCESSED_DIR / "reloop_cached_features.npz"
    if cache_path.exists():
        print(f"Loading cached representations from {cache_path}...")
        raw = np.load(cache_path)
        out = {}
        for split in ["train", "val", "test"]:
            out[split] = {
                "y_true": raw[f"{split}_y_true"],
                "logits_v0": raw[f"{split}_logits_v0"],
                "feats_v0": raw[f"{split}_feats_v0"],
                "logits_mv": raw[f"{split}_logits_mv"],
                "feats_mv": raw[f"{split}_feats_mv"],
                "mat_desc": raw[f"{split}_mat_desc"],
            }
        return out

    print("Extracting 2-view features + 12-d material descriptors (will cache to disk)...")
    views = get_two_view_transforms(IMG_SIZE)
    captured = []

    def _hook(module, inp, out):
        captured.append(out.flatten(1).detach().cpu())

    handle = model.avgpool.register_forward_hook(_hook)
    save_dict = {}
    out = {}

    for split in ["train", "val", "test"]:
        t_s = time.time()
        ds_raw = WasteDataset(split=split, transform=None)
        mat_descs = np.stack(
            [extract_12d_material_descriptors(Image.open(r["path"])) for _, r in ds_raw.data.iterrows()]
        )

        logits_v, feats_v = [], []
        y_true = None
        for v_idx, tfm in enumerate(views):
            ds = WasteDataset(split=split, transform=tfm)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
            captured.clear()
            v_l, v_y = [], []
            for images, labels in loader:
                images = images.to(device)
                v_l.append(model(images).cpu().numpy())
                if v_idx == 0:
                    v_y.append(labels.numpy())
            logits_v.append(np.concatenate(v_l, axis=0))
            feats_v.append(torch.cat(captured, dim=0).numpy())
            if v_idx == 0:
                y_true = np.concatenate(v_y, axis=0)

        logits_mv = 0.60 * logits_v[0] + 0.40 * logits_v[1]
        feats_mv = 0.60 * feats_v[0] + 0.40 * feats_v[1]

        out[split] = {
            "y_true": y_true,
            "logits_v0": logits_v[0],
            "feats_v0": feats_v[0],
            "logits_mv": logits_mv,
            "feats_mv": feats_mv,
            "mat_desc": mat_descs,
        }
        for k, val in out[split].items():
            save_dict[f"{split}_{k}"] = val
        print(f"  Split '{split}' ({len(y_true)} imgs) extracted in {time.time() - t_s:.1f}s")

    handle.remove()
    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved feature cache to {cache_path}")
    return out


@torch.no_grad()
def extract_ood_features(
    model: torch.nn.Module,
    samples: list[tuple[Image.Image, str]],
    device: torch.device,
) -> dict:
    views = get_two_view_transforms(IMG_SIZE)
    captured = []

    def _hook(module, inp, out):
        captured.append(out.flatten(1).detach().cpu())

    handle = model.avgpool.register_forward_hook(_hook)
    logits_v, feats_v = [], []
    for tfm in views:
        captured.clear()
        tensors = torch.stack([tfm(im) for im, _ in samples]).to(device)
        out_l = []
        for i in range(0, len(tensors), 64):
            out_l.append(model(tensors[i : i + 64]).cpu().numpy())
        logits_v.append(np.concatenate(out_l, axis=0))
        feats_v.append(torch.cat(captured, dim=0).numpy())
    handle.remove()

    mat_desc = np.stack([extract_12d_material_descriptors(im) for im, _ in samples])
    cats = [c for _, c in samples]
    return {
        "logits_v0": logits_v[0],
        "feats_v0": feats_v[0],
        "logits_mv": 0.60 * logits_v[0] + 0.40 * logits_v[1],
        "feats_mv": 0.60 * feats_v[0] + 0.40 * feats_v[1],
        "mat_desc": mat_desc,
        "categories": cats,
    }


def build_and_evaluate_open_set_detector(
    train_known: dict,
    val_known: dict,
    test_known: dict,
    ood_train: dict,
    ood_val: dict,
    ood_test: dict,
    calibrated: dict,
) -> tuple[dict, dict]:
    """Fits a principled Open-Set OOD / Stone / Anomaly Discriminator on:
      - Positive class (1): Train Known Waste (1,768 samples)
      - Negative class (0): Disjoint Train OOD Reference Samples (160 samples)
    Using a compact 18-dimensional Open-Set Feature Vector:
      - [0]: Max Cosine Similarity to the 6 Training Waste Class Prototypes
      - [1]: Second-Highest Prototype Cosine Similarity
      - [2]: Calibrated Top-1 Softmax Probability
      - [3]: Calibrated Top-1 vs Top-2 Probability Margin
      - [4]: Normalized Logit Energy Score
      - [5]: Negative Class-Conditional Diagonal Mahalanobis Distance in 2048-d space
      - [6..17]: 12-d Physical Material & Image Quality Descriptors
    Tunes thresholds strictly on (Val Known [379] + Val OOD [120]), then evaluates
    once on (Held-Out Test Known [380] + Held-Out Test OOD [120]).
    """
    # Compute per-class mean and variance in L2-normalized 2048-d space on Train Known
    X_tr_norm = train_known["feats_mv"] / (
        np.linalg.norm(train_known["feats_mv"], axis=1, keepdims=True) + 1e-12
    )
    y_tr = train_known["y_true"]
    cls_means = np.zeros((len(CLASS_NAMES), X_tr_norm.shape[1]), dtype=np.float32)
    cls_vars = np.zeros((len(CLASS_NAMES), X_tr_norm.shape[1]), dtype=np.float32)
    for c in range(len(CLASS_NAMES)):
        vecs = X_tr_norm[y_tr == c]
        cls_means[c] = np.mean(vecs, axis=0)
        cls_vars[c] = np.var(vecs, axis=0) + 1e-4

    def _make_18d_openset_features(split_dict: dict) -> np.ndarray:
        preds, probs, max_sim, energy = predict_with_calibrated_head(
            split_dict, calibrated, use_multiview=True
        )
        feats_norm = split_dict["feats_mv"] / (
            np.linalg.norm(split_dict["feats_mv"], axis=1, keepdims=True) + 1e-12
        )
        proto_sims = np.sort(feats_norm @ calibrated["prototypes"].T, axis=1)
        sim1 = proto_sims[:, -1]
        sim2 = proto_sims[:, -2]

        sorted_p = np.sort(probs, axis=1)
        p1 = sorted_p[:, -1]
        margin = sorted_p[:, -1] - sorted_p[:, -2]
        norm_energy = np.tanh(energy / 10.0)

        # Diagonal Mahalanobis distance to predicted class c
        maha_dist = np.zeros(len(preds), dtype=np.float32)
        for i, c in enumerate(preds):
            diff = feats_norm[i] - cls_means[c]
            maha_dist[i] = -float(np.mean((diff * diff) / cls_vars[c]))

        base_6 = np.stack([sim1, sim2, p1, margin, norm_energy, maha_dist], axis=1)
        return np.concatenate([base_6, split_dict["mat_desc"]], axis=1)

    X_os_train_pos = _make_18d_openset_features(train_known)
    X_os_train_neg = _make_18d_openset_features(ood_train)
    X_os_train = np.concatenate([X_os_train_pos, X_os_train_neg], axis=0)
    y_os_train = np.concatenate(
        [np.ones(len(X_os_train_pos)), np.zeros(len(X_os_train_neg))], axis=0
    )

    # Standardize features using Train statistics only
    feat_mean = np.mean(X_os_train, axis=0)
    feat_std = np.std(X_os_train, axis=0) + 1e-6
    X_os_train_std = (X_os_train - feat_mean) / feat_std

    ood_clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    ood_clf.fit(X_os_train_std, y_os_train)

    def _predict_knownness_prob(split_dict: dict) -> np.ndarray:
        X_raw = _make_18d_openset_features(split_dict)
        X_std = (X_raw - feat_mean) / feat_std
        probs_known = ood_clf.predict_proba(X_std)[:, 1]
        # Enforce hard image quality gate on extreme darkness/glare/flatness
        mat = split_dict["mat_desc"]
        for i in range(len(probs_known)):
            mean_luma_255 = mat[i, 0] * 255.0
            std_luma_128 = mat[i, 1] * 128.0
            if mean_luma_255 < 18.0 or (mean_luma_255 > 246.0 and std_luma_128 < 7.0) or std_luma_128 < 6.0:
                probs_known[i] = 0.0
        return probs_known

    val_k_prob = _predict_knownness_prob(val_known)
    val_u_prob = _predict_knownness_prob(ood_val)

    # Select threshold tau on Validation that keeps Known Waste FRR <= 5.5% while maximizing OOD rejection
    best_tau = 0.50
    best_j = -1.0
    for tau in np.linspace(0.20, 0.85, 66):
        tpr_k = float(np.mean(val_k_prob >= tau))
        tnr_u = float(np.mean(val_u_prob < tau))
        if (1.0 - tpr_k) <= 0.055:
            j = tpr_k + tnr_u - 1.0
            if j > best_j:
                best_j = j
                best_tau = float(round(tau, 4))

    tau_uncertain = float(round(min(0.92, best_tau + 0.08), 4))

    # Held-Out Test Evaluation
    test_k_prob = _predict_knownness_prob(test_known)
    test_u_prob = _predict_knownness_prob(ood_test)

    y_test_bin = np.concatenate([np.ones_like(test_k_prob), np.zeros_like(test_u_prob)])
    s_test_all = np.concatenate([test_k_prob, test_u_prob])
    test_auroc = float(roc_auc_score(y_test_bin, s_test_all))

    cats = np.array(ood_test["categories"])
    stone_mask = cats == "stone_rock"

    ood_metrics = {
        "tuned_knownness_threshold_tau": best_tau,
        "tuned_uncertainty_threshold_tau": tau_uncertain,
        "val_known_acceptance_rate": float(np.mean(val_k_prob >= best_tau)),
        "val_ood_rejection_rate": float(np.mean(val_u_prob < best_tau)),
        "test_auroc_known_vs_ood": test_auroc,
        "test_unknown_rejection_rate_tpr": float(np.mean(test_u_prob < best_tau)),
        "test_stone_specific_rejection_rate": float(np.mean(test_u_prob[stone_mask] < best_tau)),
        "test_false_acceptance_rate_far": float(np.mean(test_u_prob >= best_tau)),
        "test_known_false_rejection_rate_frr": float(np.mean(test_k_prob < best_tau)),
        "test_ood_breakdown_by_type": {
            c_name: float(np.mean(test_u_prob[cats == c_name] < best_tau))
            for c_name in sorted(set(ood_test["categories"]))
        },
    }

    ood_detector_params = {
        "cls_means": cls_means,
        "cls_vars": cls_vars,
        "feat_mean": feat_mean.astype(np.float32),
        "feat_std": feat_std.astype(np.float32),
        "ood_coef": ood_clf.coef_.astype(np.float32),
        "ood_intercept": ood_clf.intercept_.astype(np.float32),
        "knownness_tau": best_tau,
        "uncertainty_tau": tau_uncertain,
    }

    return ood_detector_params, ood_metrics


def main():
    set_seed(RANDOM_SEED)
    device = get_device()
    audit_summary = audit_dataset_and_splits()

    checkpoint_path = MODELS_DIR / "resnet50_best.pt"
    model = build_transfer_model("resnet50", num_classes=len(CLASS_NAMES), pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    splits_data = load_or_extract_all_features(model, device)
    train_repr = splits_data["train"]
    val_repr = splits_data["val"]
    test_repr = splits_data["test"]

    # Baseline metrics
    base_val_probs = softmax_np(val_repr["logits_v0"], temp=1.0)
    baseline_val_metrics = compute_metrics_dict(
        val_repr["y_true"], np.argmax(base_val_probs, axis=1), base_val_probs
    )
    base_test_probs = softmax_np(test_repr["logits_v0"], temp=1.0)
    baseline_test_metrics = compute_metrics_dict(
        test_repr["y_true"], np.argmax(base_test_probs, axis=1), base_test_probs
    )

    # Calibrated Improved Classifier
    calibrated = train_and_tune_improved_classifier(train_repr, val_repr)
    imp_val_preds, imp_val_probs, _, _ = predict_with_calibrated_head(val_repr, calibrated, True)
    improved_val_metrics = compute_metrics_dict(val_repr["y_true"], imp_val_preds, imp_val_probs)

    imp_test_preds, imp_test_probs, _, _ = predict_with_calibrated_head(test_repr, calibrated, True)
    improved_test_metrics = compute_metrics_dict(
        test_repr["y_true"], imp_test_preds, imp_test_probs
    )

    # Generate disjoint OOD Train (seed 55), OOD Val (seed 101), and Held-Out OOD Test (seed 202)
    ood_train_samples = generate_ood_dataset(seed=55, n_per_category=40)
    ood_val_samples = generate_ood_dataset(seed=101, n_per_category=30)
    ood_test_samples = generate_ood_dataset(seed=202, n_per_category=30)

    ood_train_repr = extract_ood_features(model, ood_train_samples, device)
    ood_val_repr = extract_ood_features(model, ood_val_samples, device)
    ood_test_repr = extract_ood_features(model, ood_test_samples, device)

    ood_detector_params, ood_metrics = build_and_evaluate_open_set_detector(
        train_repr,
        val_repr,
        test_repr,
        ood_train_repr,
        ood_val_repr,
        ood_test_repr,
        calibrated,
    )

    print("\n=== UPGRADED OPEN-SET OOD & STONE REJECTION RESULTS ===")
    print(json.dumps(ood_metrics, indent=2))

    reloop_bundle_path = MODELS_DIR / "reloop_resnet50_calibrated.pt"
    torch.save(
        {
            "backbone_state_dict": state_dict,
            "class_names": CLASS_NAMES,
            "clf_weights": torch.from_numpy(calibrated["clf_weights"]),
            "clf_intercept": torch.from_numpy(calibrated["clf_intercept"]),
            "prototypes": torch.from_numpy(calibrated["prototypes"]),
            "calibration_params": calibrated["params"],
            "ood_detector": ood_detector_params,
            "ood_thresholds": {
                "knownness_tau": ood_metrics["tuned_knownness_threshold_tau"],
                "uncertainty_tau": ood_metrics["tuned_uncertainty_threshold_tau"],
            },
        },
        reloop_bundle_path,
    )

    full_eval_out = RESULTS_DIR / "reloop_overhaul_evaluation.json"
    with open(full_eval_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_audit": audit_summary,
                "baseline_val_metrics": baseline_val_metrics,
                "baseline_test_metrics": baseline_test_metrics,
                "improved_val_metrics": improved_val_metrics,
                "improved_test_metrics": improved_test_metrics,
                "calibration_params": calibrated["params"],
                "ood_rejection_metrics": ood_metrics,
                "checkpoint_bundle": str(reloop_bundle_path.relative_to(PROJECT_ROOT)),
            },
            f,
            indent=2,
        )
    print(f"Updated {reloop_bundle_path} and {full_eval_out}")


if __name__ == "__main__":
    main()
