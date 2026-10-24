# ReLoop — Smart Campus Waste-to-Value Ecosystem

**Waste. Reward. Repeat.**

ReLoop is an AI-powered Reverse Vending Machine (RVM) software ecosystem and student reward platform built for college campuses. Students deposit campus waste into ReLoop RVM kiosks, where a calibrated **ResNet50 + 18-D Open-Set Anomaly Detection** computer vision pipeline classifies waste into 6 supported classes (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`), routes physical sorting flaps via ESP32 servo control (`60°`, `120°`, `180°`), rejects foreign/non-waste objects (such as stones or hands) via safety hold (`90°`), and issues signed QR reward tokens redeemable for canteen discounts and campus perks.

---

## Key Features

- **Calibrated Computer Vision & Open-Set OOD Rejection:**
  - **94.71% Test Accuracy** (0.9469 Macro F1) across 6 waste categories (`2,527` dataset images).
  - **99.97% OOD AUROC** & **96.67% Stone Rejection Rate** using an 18-dimensional Open-Set Discriminator (combining temperature-scaled energy, normalized Shannon entropy, class-conditional Mahalanobis distances, cosine similarity, and HSV/saturation/Laplacian texture features).
- **Interactive RVM Kiosk Touch Interface (`RVM Machine` Tab):**
  - Live WebRTC camera stream with auto-scan option, manual photo upload, and real-time sample testing (`Plastic`, `Paper`, `Metal`, `Trash`, and `Stone (OOD)`).
  - Real-time animated mechanical sorting blueprint showing the active ESP32 serial command (`BIN:RECYCLABLE:120`, `BIN:BIODEGRADABLE:60`, `BIN:NON_RECYCLABLE:180`, `REJECT:OOD:90`).
  - Multi-item disposal session queue with cryptographic QR ticket generation.
- **Student Companion App (`Student App` Tab):**
  - Built-in QR scanner (`jsQR`) + 1-click instant session redemption.
  - Dynamic reward pricing (`Metal`: 20 pts, `Glass`: 18 pts, `Plastic`: 15 pts, `Cardboard`: 12 pts, `Paper`: 10 pts, `Trash`: 5 pts).
  - Canteen reward store, hostel leaderboard, and live campus RVM kiosk capacity finder.

---

## Project Architecture

```text
NSS_Sociothon/
├── index.html                        # ReLoop Web Application (RVM Kiosk + Student App)
├── styles.css                        # ReLoop Eco-Industrial Design System
├── app.js                            # Frontend Logic, Camera/Upload, QR Generation & Scanning
├── server.py                         # Unified HTTP & REST API Server (/api/predict, /api/deposit, etc.)
├── requirements.txt                  # Python Dependencies
├── HARDWARE_AND_WORKFLOW_DOCS.md     # Hardware BOM, ESP32 Wiring & Complete Architecture Docs
├── hardware_and_ai/
│   ├── rvm_ai_engine.py              # Production PyTorch ResNet50 + 18-D Open-Set Inference Engine
│   └── esp32_rvm_controller.ino      # ESP32 Firmware for Ultrasonic Sensor, Servos & Load Cell
├── tests/
│   └── test_reloop_system.py         # End-to-End Automated Test Suite
└── waste-classification-ai-main/
    ├── models/
    │   ├── model_metadata.json       # Model Architecture & Normalization Config
    │   └── reloop_ood_calibration.json # Temperature (T=0.6929), Centroids & OOD Classifier Params
    ├── reports/
    │   ├── reloop_audit_and_evaluation.md
    │   └── results/reloop_overhaul_evaluation.json
    └── src/
        └── training/
            ├── reloop_pipeline.py        # Feature Extraction, Temp Scaling & Head Fine-Tuning
            └── upgrade_ood_detector.py   # 18-D Open-Set Discriminator Training & Evaluation
```

---

## Quick Start

1. **Install Dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Start the ReLoop Server:**
   ```bash
   python server.py
   ```

3. **Open in Browser:**
   Navigate to `http://localhost:8080` to use the **RVM Machine** and **Student App** interfaces.

4. **Run End-to-End Tests:**
   ```bash
   python -m unittest tests/test_reloop_system.py -v
   ```
