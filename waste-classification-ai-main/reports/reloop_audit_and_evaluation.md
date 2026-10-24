# ReLoop — AI Model Audit, Calibration, Open-Set OOD Rejection & Evaluation Report

## 1. Root-Cause Audit of Previous Classification Failures

Before modifying the application, a complete end-to-end audit of the repository (`waste-classification-ai-main`, `hardware_and_ai/`, and the web frontend) identified four root causes explaining why uploaded images and live-camera frames previously produced inaccurate or random predictions:

1. **Frontend Bypassed the Trained PyTorch Model and Used a Naive RGB Heuristic with a Random Fallback:**
   - In the previous `app.js`, `handleWasteImageUpload()` and `captureFromLiveCamera()` did **not** invoke the trained PyTorch `ResNet50` model (`models/resnet50_best.pt`).
   - Instead, `analyzeCanvasImageFeatures()` averaged raw red, green, and blue pixel values across the $224 \times 224$ canvas and fell back to `return WASTE_CATALOG[Math.floor(Math.random() * 5)]` whenever simple color thresholds were not met.
2. **Static File Server Instead of an Active Model Inference Backend:**
   - The web application was served via `python -m http.server 8080`, which only served static files and could not execute PyTorch inference.
3. **Non-Portable Hardcoded Paths in `data/processed/splits.csv`:**
   - Every row in `splits.csv` referenced `C:\Users\Martin\Downloads\Amit Data Science\projects\Deep Learning Project\data\raw\dataset-resized\...`, causing `WasteDataset` to fail on any other machine unless paths were resolved relative to the local `TRASHNET_DIR`.
4. **Absence of Class-Prior Calibration, Image-Quality Filtering, and Open-Set / Out-of-Distribution (OOD) Rejection:**
   - During Stage B fine-tuning of `resnet50_best.pt`, inverse-frequency class weights (`weight_trash = 3.069` vs. `weight_paper = 0.708`) heavily shifted the final linear layer bias toward `trash`. While `trash` recall reached `95.00%`, `trash` test precision dropped to **`63.33%`** (`F1 = 0.7600`), pulling ambiguous `paper`, `plastic`, and `cardboard` images into `trash`.
   - Furthermore, the closed-set 6-class softmax layer had no image-quality gate (for dark, blurry, or empty frames) and no open-set rejection layer for unsupported objects such as **stones**.

---

## 2. Dataset Audit & Split Integrity Verification

- **Dataset:** TrashNet resized (`512 × 384` RGB `.jpg` images across 6 classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`).
- **Total Verified Images:** `2,527 / 2,527` (`0` missing files, `0` corrupted files).
- **Stratified Split (`seed = 42`):**
  - **Train (70%):** `1,768` images (`cardboard: 282`, `glass: 350`, `metal: 287`, `paper: 416`, `plastic: 337`, `trash: 96`)
  - **Validation (15%):** `379` images (`cardboard: 61`, `glass: 75`, `metal: 61`, `paper: 89`, `plastic: 72`, `trash: 21`)
  - **Held-Out Test (15%):** `380` images (`cardboard: 60`, `glass: 76`, `metal: 62`, `paper: 89`, `plastic: 73`, `trash: 20`)
- **Cross-Split Leakage Check (MD5 File Hash Audit):**
  - `train ∩ test = 0` exact duplicates (`0.00%` test leakage).
  - `val ∩ test = 0` exact duplicates (`0.00%` test leakage).
  - `train ∩ val = 2` exact duplicates in the upstream TrashNet archive.

---

## 3. Known-Class Model Improvement (Baseline vs. Improved Calibrated ResNet50)

### Methodology
1. **Strict Split Discipline:** All classifier weights ($\mathbf{W}_{\text{head}}, \mathbf{b}_{\text{head}}$) and L2-normalized 2048-d class prototypes ($\boldsymbol{\mu}_c$) were fitted strictly on the **Training Split (`1,768` images)**.
2. **Validation-Driven Hyperparameter & Prior Tuning:** Blending weights ($\alpha_{\text{head}} = 0.70$), `trash` class-prior bias correction ($\Delta b_{\text{trash}} = -0.35$), and temperature scaling ($T = 0.85$) were selected strictly on the **Validation Split (`379` images)**.
3. **Multi-View Inference Aggregation:** Combines deterministic evaluation views (`0.60 × Resize(224, 224)` + `0.40 × HorizontalFlip(224, 224)`) in the 2048-d penultimate feature space (`avgpool`).
4. **Single Held-Out Test Evaluation:** Evaluated once on the **Held-Out Test Split (`380` images)**.

### Official Baseline vs. Improved Results

| Split & Metric | Baseline `resnet50_best.pt` | Improved `reloop_resnet50_calibrated.pt` | Change |
| :--- | :---: | :---: | :---: |
| **Validation Loss (379 imgs)** | 0.3272 | **0.2801** | **-0.0471** |
| **Validation Accuracy (379 imgs)** | 87.86% | **89.18%** | **+1.32%** |
| **Validation Macro-F1** | 0.8554 | **0.8753** | **+0.0199** |
| **Held-Out Test Loss (380 imgs)** | 0.3069 | **0.2554** | **-0.0515** |
| **Held-Out Test Accuracy (380 imgs)** | 91.05% (346/380) | **92.11% (350/380)** | **+1.06%** |
| **Held-Out Test Macro-Precision** | 88.45% | **92.52%** | **+4.07%** |
| **Held-Out Test Macro-Recall** | 91.60% | **92.01%** | **+0.41%** |
| **Held-Out Test Macro-F1** | 0.8941 | **0.9216** | **+0.0275** |
| **Held-Out Test Weighted-F1** | 0.9127 | **0.9210** | **+0.0083** |

### Per-Class Held-Out Test Performance (380 Images)

| Class | Support | Baseline Precision | Baseline Recall | Baseline F1 | Improved Precision | Improved Recall | Improved F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cardboard** | 60 | 96.43% | 90.00% | 0.9310 | **96.55%** | **93.33%** | **0.9492** |
| **Glass** | 76 | 92.11% | 92.11% | 0.9211 | 92.00% | 90.79% | 0.9139 |
| **Metal** | 62 | 88.06% | 95.16% | 0.9147 | 85.92% | **98.39%** | **0.9173** |
| **Paper** | 89 | 95.29% | 91.01% | 0.9310 | 93.26% | **93.26%** | **0.9326** |
| **Plastic** | 73 | 95.45% | 86.30% | 0.9065 | 92.65% | 86.30% | 0.8936 |
| **Trash** | 20 | 63.33% | 95.00% | 0.7600 | **94.74%** | 90.00% | **0.9231** |

---

## 4. Open-Set / Unknown-Object (OOD & Stone) Rejection Evaluation

To prevent ReLoop from forcing unsupported objects (such as **stones**, foliage, non-waste patterns, or dark/blurry frames) into recycling bins, `ReLoopVisionEngine` (`hardware_and_ai/rvm_ai_engine.py`) implements a multi-stage verification pipeline:

1. **Stage 1 — Image Quality & Presence Gate (`POOR_QUALITY`):**
   - Rejects underexposed/dark images (`brightness_mean < 18`), overexposed glare (`brightness_mean > 247` with low contrast), empty uniform backgrounds (`contrast_std < 6.5`), and blurry images (`laplacian_var < 10.5`).
2. **Stage 2 — 18-Dimensional Open-Set Discriminator (`UNKNOWN`):**
   - Combines 6 deep representation signals from ResNet50's 2048-d `avgpool` layer (Top-1 & Top-2 Prototype Cosine Similarity, Calibrated Top-1 Confidence, Top-1/Top-2 Margin, Normalized Logit Energy, and Class-Conditional Diagonal Mahalanobis Distance) with 12 physical material and texture descriptors (chromatic saturation mean/std, specular highlight tail $p_{99}-p_{90}$, Laplacian sharpness, micro-speckle-to-gradient ratio, and opponent color variance).
   - Fitted on `Train Known (1,768)` vs. a disjoint reference negative set (`seed=55`), tuned on `Val Known (379) + Val OOD (120, seed=101)` (`tau_ood = 0.20`, `tau_uncertain = 0.28`), and evaluated on `Held-Out Test Known (380) + Held-Out Test OOD (120, seed=202)`.

### Held-Out Open-Set Rejection Metrics

| Metric | Measured Value |
| :--- | :---: |
| **Held-Out Known vs. OOD AUROC** | **0.9997** |
| **Overall Unknown-Object Rejection Rate (TPR on OOD)** | **99.17%** (119 / 120) |
| **Stone / Rock Specific Rejection Rate** | **96.67%** (29 / 30) |
| **Natural Foliage / Non-Waste Pattern / Poor Quality Rejection** | **100.00%** (90 / 90) |
| **Known-Waste False Rejection Rate (FRR on 380 Test Images)** | **0.53%** (2 / 380) |
| **Unknown-Object False Acceptance Rate (FAR on 120 OOD Images)** | **0.83%** (1 / 120) |
